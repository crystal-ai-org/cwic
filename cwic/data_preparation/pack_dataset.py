
import datasets
import numpy as np
from tqdm import tqdm
from transformers import LlamaTokenizerFast


TOKENIZER = "pad_tokenizer"

READ_URL = "aklein4/fineweb-edu-sample-10BT-shuffled"
WRITE_URL = "aklein4/fineweb-edu-sample-10BT-llama3"

BATCH_SIZE = 2048

MAX_LENGTH = 1024
MIN_LENGTH = 1024 - 64

Q_SIZE = 8192


class TokenPackingQueue:
   
    def __init__(self, max_length, min_length, q_size=Q_SIZE):

        self.q_size = q_size
        self.max_length = max_length
        self.min_length = min_length
    
        self.queue = [None] * q_size
        self.id_queue = [None] * q_size
        self.counts = np.array([0] * q_size)
        self.filled = np.array([False] * q_size)

        self.total_count = 0
        self.trunc_count = 0


    def _pop(self, ind):
        assert self.queue[ind] is not None
        assert self.id_queue[ind] is not None
        assert self.filled[ind]

        x = self.queue[ind]
        ids = self.id_queue[ind]

        self.queue[ind] = None
        self.id_queue[ind] = None
        self.counts[ind] = 0
        self.filled[ind] = False   

        return x, ids


    def _push(self, x, ids, ind):
        assert self.queue[ind] is None
        assert self.id_queue[ind] is None
        assert self.counts[ind] == 0
        assert not self.filled[ind]

        self.queue[ind] = x
        self.id_queue[ind] = ids
        self.counts[ind] = x.size
        self.filled[ind] = True 


    def __call__(self, x):

        # truncate to max length
        x = x[:self.max_length]

        # ids init to zero
        ids = np.zeros_like(x)

        # only work with small sequences
        if x.size < self.min_length:

            # try to combine from queue
            new_sizes = self.counts + x.size
            good = np.logical_and(self.filled, new_sizes <= self.max_length)
            if np.any(good):

                # get the biggest combination
                new_sizes[~good] = -1
                ind = np.argmax(new_sizes)
                
                # get new x
                x = np.concatenate(
                    [self.queue[ind], x],
                    axis=-1
                )
                ids = np.concatenate(
                    [self.id_queue[ind], ids+self.id_queue[ind][-1]+1],
                    axis=-1
                )

                # pop from queue
                self._pop(ind)

            # still not big enough, add to queue
            if x.size < self.min_length:

                # pop the largest from queue if full
                if self.filled.all():
                    max_ind = np.argmax(self.counts)

                    y, y_ids = self._pop(max_ind)
                else:
                    y, y_ids = None, None
                
                # push to queue
                avail = np.argmin(self.filled)
                self._push(x, ids, avail)

                # restore possibly popped
                x, ids = y, y_ids

        if x is not None:
            assert x.size <= self.max_length
            assert ids.shape == x.shape

            self.total_count += self.max_length
            self.trunc_count += x.size

        return x, ids


class TokenizerMap:

    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
    

    def __call__(self, d):
        
        # batch encode text
        input_ids = self.tokenizer(
            [t[:20*self.max_length] for t in d["text"]],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np"
        ).input_ids
        
        input_ids = input_ids.astype(np.uint32)
        
        # convert to list
        out = []
        for curr in input_ids:
            out.append(curr[curr != self.tokenizer.pad_token_id])

        return {"input_ids": out}


def main():
    
    tokenizer = LlamaTokenizerFast.from_pretrained(TOKENIZER)

    data = datasets.load_dataset(READ_URL, split="train", streaming=False)
    
    data = data.map(
        TokenizerMap(tokenizer, MAX_LENGTH),
        batched=True,
        batch_size=BATCH_SIZE,
        remove_columns=data.column_names,
    )

    input_ids = []
    segment_ids = []
    queue = TokenPackingQueue(MAX_LENGTH, MIN_LENGTH, Q_SIZE)

    with tqdm(data, desc="Packing Queue") as pbar:
        for d in pbar:

            x, ids = queue(np.array(d["input_ids"]).astype(np.uint32))

            if x is not None:
                assert np.max(x) < tokenizer.vocab_size+1000
                assert np.max(ids) < 1000

                input_ids.append(x)
                segment_ids.append(ids)
        
            pbar.set_postfix({
                "total": f"{queue.total_count:_}",
                "trunc": f"{queue.trunc_count:_}",
                "frac": f"{100*queue.trunc_count/(1+queue.total_count):.3f}%",
            })

    data = datasets.concatenate_datasets(
        [
            datasets.Dataset.from_dict({
                "input_ids": input_ids[::100],
                "segment_ids": segment_ids[::100],
            }) for i in range(100)
        ]
    )
    data = data.shuffle(seed=42)

    data.push_to_hub(WRITE_URL, private=False)
    tokenizer.push_to_hub(WRITE_URL, private=False, subfolder="tokenizer")


if __name__ == "__main__":
    main()
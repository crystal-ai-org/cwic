import re
import torch

import numpy as np
import matplotlib.pyplot as plt
from html import escape as html_escape
import os
from scipy.stats import norm

from transformers import pipeline

from cwic_huggingface.modelling_cwic import CWICForCausalLM

import matplotlib as mpl

OUTPUT_FOLDER = "./flop_diagrams"
FR = 6 # Nx active parameter reduction over Llama-3.1-1B-Instruct
# NORMALIZER for flop coloring in terminal as well
CHECKPOINT = {
    3: "crystal-ai/CWICLlama-3.2-1B-A413M-Instruct",
    6: "crystal-ai/CWICLlama-3.2-1B-A206M-Instruct",
}[FR]

from termcolor import colored, cprint


def show_flops(tokenizer, input_ids, flops):

    # med, mx = np.median(flops), flops.max()

    # colors = 3 * (flops - med) / (mx - med + 1e-7)
    # colors = norm.cdf(colors)
    # plt.plot(colors)
    # plt.savefig("color_debug.png")
    # plt.clf()
    # colors = flops / flops.max()

    # colors = flops / flops.mean()
    # colors = np.log10(colors)
    # colors = colors - colors.mean()
    # colors = 0.5 * colors / (np.std(colors) + 0.0001)
    # colors = np.argsort(np.argsort(colors)) / colors.shape[0]
    
    colors = flops* FR / 2 # / flops.max()
    colors = colors.tolist()
    def terminal_escape(x):
        return x
    
    cmap = mpl.colormaps['YlGnBu']
    cmap_fn = lambda x: str(tuple(max(min(int(c* 255),255),0) for c in cmap(x*0.5)[:3]))
    def f(m,x):
        
        text=tokenizer.decode([m])
        outs=[]
        for t in text: #re.findall(u'(?:[\ud800-\udbff][\udc00-\udfff])|.',text):
            esc=terminal_escape(t)
            if t=="\n":
                outs.append(esc)
            else:
                outs.append(colored(esc,(0,0,0),tuple(max(min(int(c* 255),255),0) for c in cmap(x*0.5))[:3]))
        return "".join(outs)

    print("".join(
            [
                f(m,x)
                for m, x in zip(
                    input_ids,
                    [0.0] + colors[:-1],
                )
            ]
        ))
    html = (
        """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,100..900;1,14..32,100..900&display=swap" rel="stylesheet">"""
        + "<pre style=\"background:white;\">"
        + "".join(
            [
                f'<span style="font-family: Inter;font-optical-sizing: auto;font-weight: normal;font-style: normal; background-color: rgb{cmap_fn(c)};">{html_escape(tokenizer.decode([m]))}</span>'
                for m, c in zip(
                    input_ids,
                    [0.0] + colors[:-1],
                )
            ]
        )
        + "</pre>"
    )

    return html


def main():
    
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    pipe = pipeline(
        "text-generation",
        model=CHECKPOINT,
        device_map="cuda:0",
        torch_dtype=torch.float32,
    )
    print("Pipeline loaded successfully.")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    ind = 0
    while True:
        
        print(f"\n=== Example {ind} ===\n")
        my_message = input("Enter your message (or 'exit' to quit): ")
        if my_message.lower().strip() == 'exit':
            break
        print("\nThinking...")

        messages = [
            # {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
            {"role": "user", "content": my_message},
        ]

        outputs = pipe(
            messages,
            max_new_tokens=512,
        )

        output_text = outputs[0]["generated_text"][-1]['content']

        messages += [
            {
                "role": "assistant",
                "content": output_text,
            }
        ]

        input_ids = pipe.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
        ).to("cuda:0")
        out = pipe.model(
            input_ids=input_ids
        )

        input_ids = input_ids[0].detach().cpu().numpy()
        flops = (out.active_parameters/out.dense_parameters)[0].detach().cpu().numpy()

        html = show_flops(pipe.tokenizer, input_ids, flops)

        with open(os.path.join(OUTPUT_FOLDER, f"flop_diagram_{ind}.html"), "w") as f:
            f.write(html)

        plt.plot(flops)
        plt.savefig(os.path.join(OUTPUT_FOLDER, f"flop_plot_{ind}.png"))
        plt.clf()

        np.save(
            os.path.join(OUTPUT_FOLDER, f"data_{ind}.npy"),
            {"input_ids": input_ids, "flops": flops}
        )

        ind += 1



if __name__ == "__main__":
    main()

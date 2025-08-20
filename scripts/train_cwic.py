import json
import os

from model_code.utils import replace_grad


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.98"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer
from jax.sharding import PartitionSpec, NamedSharding
from flax import nnx
from jax import Array, numpy as jnp
import jax
import math
import tqdm
import wandb
import optax
import datetime
import datasets
import numpy as np
import orbax.checkpoint as ocp
from functools import partial
import argparse

from model_code.CWICMod import CWICDense, CWICFFN
from model_code.loss_names import LossKeys
from model_code.optimizers import clip_by_global_norm_quantile


from model_code.nnx_model import (
    NNXRefractionModule,
    CWICConfig,
    infer_nnx_model,
)
from model_code.logical_axis import global_mesh
from model_code.config import CWICConfig

parser = argparse.ArgumentParser(description="Test.")
parser.add_argument(
    "--restore_path",
    action="store",
    type=str,
    required=False,
    help="The path from which we are restoring.",
    default="NO_RUN",
)
parser.add_argument("--name", action="store", type=str, help="Name for the run in wandb")
parser.add_argument(
    "--notes",
    action="store",
    type=str,
    required=False,
    default=None,
    help="Notes for the run in wandb, similar to a git commit.",
)
parser.add_argument(
    "--tags",
    action="store",
    type=str,
    required=False,
    default=None,
    help="Tags to attach to the wandb run, seperated by commas.",
)
parser.add_argument(
    "--save_checkpoints_to",
    action="store",
    type=str,
    required=False,
    default=None,
    help="save checkpoints to this local path or gs:// url",
)

args = parser.parse_args()
save_ckpts = args.save_checkpoints_to is not None
local_ckpts = save_ckpts and not args.save_checkpoints_to.startswith("gs://")
if not save_ckpts:
    print("WARNING: --save_checkpoints_to is not set, not saving checkpoints")


# IMPORTANT: due to the way the flop penalty works over a whole gradient batch:
# increasing grad_accums is NOT the same as a larger fulll gradient batch size with grad_accums=1, (use more gpus)
# the larger full gradient batch size with perform BETTER
# ye been warned!
# our runs were trained with 128 batch size across devices but doing more would be awesome

checkpoint, est_p_count, grad_batch_dim_per_proc = [
    ("unsloth/Llama-3.2-1B-Instruct", 1000000000, 1),
    ("unsloth/Llama-3.2-3B-Instruct", 3000000000, 8),
][0]
grad_batch_dim_per_proc *= jax.local_device_count()

dataset_name = "crystal-ai/chat-compilation-Llama-3.2-Instruct-Shuffled"



tokenizer = AutoTokenizer.from_pretrained(checkpoint)

hf_model = AutoModelForCausalLM.from_pretrained(checkpoint).to("cpu")

print("This is the checkpoint:", checkpoint)
print("These are the model internals:", hf_model.model.layers)

og_model_config = hf_model.model.config
og_model_config.initializer_range = og_model_config.hidden_size**-0.5

mesh = global_mesh

import configparser

iniconfig = configparser.ConfigParser()
iniconfig.read("./runconfig.ini")
config = CWICConfig(
    **vars(og_model_config),
    cwic_stripe_size=eval(iniconfig["ablations"]["stripe_size"]),
    cwic_stripe_size_lm_head=eval(iniconfig["ablations"]["stripe_size_lm_head"]),
    cwic_threshold_learning_scale=40.0,
    cwic_bandwidth=0.1,
    cwic_threshold_shift_cap=1.0,
)
with mesh:

    @nnx.jit
    def make_sharded_teacher():
        model = NNXRefractionModule(
            config=config,
            insane=False,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            rngs=nnx.Rngs(0),
            # precision='fastest'
        )
        state = nnx.state(model)
        pspecs = nnx.get_partition_spec(state)  # Strip out the annotations from state.
        sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
        nnx.update(model, sharded_state)  # The model is sharded now!
        return model

    teacher_model = make_sharded_teacher()

assert isinstance(teacher_model.h.blocks.attention.wq, nnx.Linear)
assert isinstance(teacher_model.h.blocks.attention.wk, nnx.Linear)
assert isinstance(teacher_model.h.blocks.attention.wv, nnx.Linear)
assert isinstance(teacher_model.h.blocks.attention.wo, nnx.Linear)

assert isinstance(teacher_model.h.blocks.feed_forward.up, nnx.Linear)
assert isinstance(teacher_model.h.blocks.feed_forward.gate, nnx.Linear)
assert isinstance(teacher_model.h.blocks.feed_forward.down, nnx.Linear)

assert isinstance(teacher_model.lm_head, nnx.Linear)

with mesh:

    @nnx.jit
    def make_sharded_student():
        model = NNXRefractionModule(
            config=config,
            insane=True,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            rngs=nnx.Rngs(0),
            # precision='fastest'
        )
        state = nnx.state(model)
        pspecs = nnx.get_partition_spec(state)  # Strip out the annotations from state.
        sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
        nnx.update(model, sharded_state)  # The model is sharded now!
        return model

    student_model = make_sharded_student()

assert isinstance(student_model.h.blocks.attention.wqkv_i, CWICDense)
assert isinstance(student_model.h.blocks.attention.wo_i, CWICDense)

assert isinstance(student_model.h.blocks.feed_forward.ffn_i, CWICFFN)

assert isinstance(student_model.lm_head, CWICDense)


# qkvo
# @nnx.jit
def copy_from_hf(teacher_model):  # The model's state, a pure pytree.
    # pspecs = nnx.get_named_sharding(state,mesh)     # Strip out the annotations from state.
    # pspecs = nnx.get_partition_spec(state)     # Strip out the annotations from state.
    teacher_model.wte.embedding.value = jax.device_put(
        jnp.array(hf_model.model.embed_tokens.weight.data.numpy(), teacher_model.param_dtype),
        teacher_model.wte.embedding.value.device,
    )
    teacher_model.h.blocks.attention.wq.kernel.value = jax.device_put(
        jnp.stack(
            [
                jnp.array(
                    block.self_attn.q_proj.weight.data.numpy().swapaxes(0, 1),
                    teacher_model.param_dtype,
                )
                for block in hf_model.model.layers
            ]
        ),
        teacher_model.h.blocks.attention.wq.kernel.value.device,
    )

    teacher_model.h.blocks.attention.wk.kernel.value = jax.device_put(
        jnp.stack(
            [
                jnp.array(
                    block.self_attn.k_proj.weight.data.numpy().swapaxes(0, 1),
                    teacher_model.param_dtype,
                )
                for block in hf_model.model.layers
            ]
        ),
        teacher_model.h.blocks.attention.wk.kernel.value.device,
    )

    teacher_model.h.blocks.attention.wv.kernel.value = jax.device_put(
        jnp.stack(
            [
                jnp.array(
                    block.self_attn.v_proj.weight.data.numpy().swapaxes(0, 1),
                    teacher_model.param_dtype,
                )
                for block in hf_model.model.layers
            ]
        ),
        teacher_model.h.blocks.attention.wv.kernel.value.device,
    )
    teacher_model.h.blocks.attention.wo.kernel.value = jax.device_put(
        jnp.stack(
            [
                jnp.array(
                    block.self_attn.o_proj.weight.data.numpy().swapaxes(0, 1),
                    teacher_model.param_dtype,
                )
                for block in hf_model.model.layers
            ]
        ),
        teacher_model.h.blocks.attention.wo.kernel.value.device,
    )

    # ffn
    teacher_model.h.blocks.feed_forward.gate.kernel.value = jax.device_put(
        jnp.stack(
            [
                jnp.array(
                    block.mlp.gate_proj.weight.data.numpy().swapaxes(0, 1),
                    teacher_model.param_dtype,
                )
                for block in hf_model.model.layers
            ]
        ),
        teacher_model.h.blocks.feed_forward.gate.kernel.value.device,
    )
    teacher_model.h.blocks.feed_forward.down.kernel.value = jax.device_put(
        jnp.stack(
            [
                jnp.array(
                    block.mlp.down_proj.weight.data.numpy().swapaxes(0, 1),
                    teacher_model.param_dtype,
                )
                for block in hf_model.model.layers
            ]
        ),
        teacher_model.h.blocks.feed_forward.down.kernel.value.device,
    )

    teacher_model.h.blocks.feed_forward.up.kernel.value = jax.device_put(
        jnp.stack(
            [
                jnp.array(
                    block.mlp.up_proj.weight.data.numpy().swapaxes(0, 1),
                    teacher_model.param_dtype,
                )
                for block in hf_model.model.layers
            ]
        ),
        teacher_model.h.blocks.feed_forward.up.kernel.value.device,
    )
    assert isinstance(teacher_model.h.blocks.attention_norm, nnx.RMSNorm)
    teacher_model.h.blocks.attention_norm.scale.value = jax.device_put(
        jnp.stack(
            [
                jnp.array(block.input_layernorm.weight.data.numpy(), teacher_model.param_dtype)
                for block in hf_model.model.layers
            ]
        ),
        teacher_model.h.blocks.attention_norm.scale.value.device,
    )
    assert isinstance(teacher_model.h.blocks.ffn_norm, nnx.RMSNorm)
    teacher_model.h.blocks.ffn_norm.scale.value = jax.device_put(
        jnp.stack(
            [
                jnp.array(
                    block.post_attention_layernorm.weight.data.numpy(), teacher_model.param_dtype
                )
                for block in hf_model.model.layers
            ]
        ),
        teacher_model.h.blocks.ffn_norm.scale.value.device,
    )

    # head
    teacher_model.lm_head.kernel.value = jax.device_put(
        jnp.array(hf_model.lm_head.weight.data.numpy().swapaxes(0, 1), teacher_model.param_dtype),
        teacher_model.lm_head.kernel.value.device,
    )

    assert isinstance(teacher_model.ln_f, nnx.RMSNorm)
    teacher_model.ln_f.scale.value = jax.device_put(
        jnp.array(hf_model.model.norm.weight.data.numpy(), teacher_model.param_dtype),
        teacher_model.ln_f.scale.value.device,
    )


jax.debug.visualize_array_sharding(teacher_model.lm_head.kernel.value)

print("PRE TEACHER<HF")
print("array_sharding(teacher_model.lm_head.kernel.value)")
jax.debug.visualize_array_sharding(teacher_model.lm_head.kernel.value)
print("array_sharding(teacher_model.h.blocks.attention.wq.kernel.value[0])")
jax.debug.visualize_array_sharding(teacher_model.h.blocks.attention.wq.kernel.value[0])

with mesh:
    copy_from_hf(teacher_model)

print("POST TEACHER<HF")
print("array_sharding(teacher_model.lm_head.kernel.value)")
jax.debug.visualize_array_sharding(teacher_model.lm_head.kernel.value)
print("array_sharding(teacher_model.h.blocks.attention.wq.kernel.value[0])")
jax.debug.visualize_array_sharding(teacher_model.h.blocks.attention.wq.kernel.value[0])


@nnx.jit
def transfer(student_model, teacher_model):

    def transfer_qkv(insane: CWICDense, q: jax.Array, k: jax.Array, v: jax.Array, rngs: nnx.Rngs):
        qkv = jnp.concatenate([q, k, v], axis=-1)
        insane.transfer_teacher(rngs, qkv)

    def transfer_single(insane: CWICDense, head: jax.Array, rngs: nnx.Rngs):
        insane.transfer_teacher(rngs, head)

    def transfer_ffn(
        insane: CWICFFN, up: jax.Array, down: jax.Array, gate: jax.Array, rngs: nnx.Rngs
    ):
        insane.transfer_teacher(rngs, up, down, gate)

    student_model.wte.embedding.value = jnp.astype(
        teacher_model.wte.embedding.value, student_model.param_dtype
    ).copy()

    transfer_rngs = nnx.Rngs(0)
    transfer_qkv(
        student_model.h.blocks.attention.wqkv_i,
        teacher_model.h.blocks.attention.wq.kernel.value,
        teacher_model.h.blocks.attention.wk.kernel.value,
        teacher_model.h.blocks.attention.wv.kernel.value,
        transfer_rngs,
    )
    transfer_single(
        student_model.h.blocks.attention.wo_i,
        teacher_model.h.blocks.attention.wo.kernel.value,
        transfer_rngs,
    )
    transfer_single(
        student_model.lm_head,
        teacher_model.lm_head.kernel.value,
        transfer_rngs,
    )
    transfer_ffn(
        student_model.h.blocks.feed_forward.ffn_i,
        teacher_model.h.blocks.feed_forward.up.kernel.value,
        teacher_model.h.blocks.feed_forward.down.kernel.value,
        teacher_model.h.blocks.feed_forward.gate.kernel.value,
        transfer_rngs,
    )

    student_model.h.blocks.attention_norm.scale.value = (
        student_model.h.blocks.attention_norm.scale.value.at[:].set(
            teacher_model.h.blocks.attention_norm.scale.value
        )
    )  # type: ignore
    student_model.h.blocks.ffn_norm.scale.value = student_model.h.blocks.ffn_norm.scale.value.at[:].set(teacher_model.h.blocks.ffn_norm.scale.value)  # type: ignore
    student_model.ln_f.scale.value = student_model.ln_f.scale.value.at[:].set(teacher_model.ln_f.scale.value)  # type: ignore
    return student_model


print("PRE STUDENT<TEACHER")
print("array_sharding(student_model.lm_head.W.kernel.value)")
jax.debug.visualize_array_sharding(student_model.lm_head.W.kernel.value)
print("array_sharding(student_model.h.blocks.attention.wqkv_i.W.kernel.value[0])")
jax.debug.visualize_array_sharding(student_model.h.blocks.attention.wqkv_i.W.kernel.value[0])

with mesh:
    student_model = transfer(student_model, teacher_model)
print("POST STUDENT<TEACHER")

print("array_sharding(student_model.lm_head.W.kernel.value)")
jax.debug.visualize_array_sharding(student_model.lm_head.W.kernel.value)
print("array_sharding(student_model.h.blocks.attention.wqkv_i.W.kernel.value[0])")
jax.debug.visualize_array_sharding(student_model.h.blocks.attention.wqkv_i.W.kernel.value[0])
print("done making a model.")
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def do_inference(messages):
    i_model = student_model
    i_model.eval()
    token_id_list = list(
        tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
    )
    new_tok_list = []
    new_tok_list_flops = []
    import jax

    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    new_infer_tok_count = 1024
    max_len = len(token_id_list) + new_infer_tok_count
    max_len = 16  # 256  # 2048  # 2**int(math.ceil(math.log2(max_len-1)))+
    if i_model.h.blocks.attention.cache_index is None:
        i_model.init_cache((1, max_len - 1, 1))
    else:
        i_model.reset_cache()

    # max_len=2**int(math.ceil(math.log2(max_len-1)))+1
    feed_tokens = token_id_list
    tots = 0
    for mm in range(len(token_id_list), max_len):
        pad_len = 2 ** int(math.ceil(math.log2(len(feed_tokens)))) + 1
        eos_pad_count = 0  # pad_len - len(feed_tokens)
        cc = 2 ** int(math.floor(math.log2(len(feed_tokens))))
        toks = jnp.array([feed_tokens[:cc]])
        feed_tokens = feed_tokens[cc:]
        msk = toks != eos_token_id
        logits, aux_infer = i_model(
            toks,
            jnp.zeros_like(toks),
            msk,
            # jnp.ones_like(toks),
            tots + jnp.cumsum(msk, axis=-1) - 1,
            jax.random.key(0),
        )
        tots += cc
        if len(feed_tokens) > 0:
            continue
        new_token = int(jnp.argmax(logits, axis=-1)[0][-1])
        if new_token == tokenizer.eos_token_id:
            break
        sta = len("".join(new_tok_list))
        token_id_list.append(new_token)
        feed_tokens.append(new_token)
        new_tok_list.append(tokenizer.decode([new_token]))
        print(new_tok_list[-1], end="", flush=True)
        ratio = float(
            jnp.ravel((aux_infer[LossKeys.FLOPS_CWIC] / aux_infer[LossKeys.FLOPS_BASE]))[-1]
        )
        if not (ratio < 1):
            ratio = 1
        new_tok_list_flops.append(
            {"start": sta, "end": len("".join(new_tok_list)), "ratio": ratio}
        )
        # for new_token in jnp.argmax(logits, axis=-1)[0]:
        #     token_id_list.append(new_token)
    print("")
    strr = "".join(new_tok_list)
    i_model.delete_cache()
    return messages + [{"role": "assistant", "content": strr, "flopSpans": new_tok_list_flops}]


print("-=-")
print("CI BEFORE", student_model.h.blocks.attention.cache_index)
print("-=-")
print(do_inference([{"role": "user", "content": "Hi"}])[-1])
print("-=-")
print("CI AFTER", student_model.h.blocks.attention.cache_index)
print("-=-")

# vocab params
bos_token_id = tokenizer.bos_token_id
eos_token_id = tokenizer.eos_token_id
assert isinstance(eos_token_id, int)
VOCAB_SIZE = og_model_config.vocab_size

max_train_seq_len = 1024

grad_batch_dim = grad_batch_dim_per_proc * jax.process_count()
grad_accums = int(eval(iniconfig["experiment"]["grad_accums"]))
batch_dim = grad_accums * grad_batch_dim
epochs = eval(iniconfig["experiment"]["epochs"])
HIST_INTERVAL: int | bool = False
DISCRETE_INTERVAL: int | bool = False
TEST_INTERVAL = 100


flop_ratio_max = eval(iniconfig["experiment"]["flop_ratio_max"])

train_schedule_length = epochs * eval(iniconfig["experiment"]["dataset_length"]) / batch_dim
flop_schedule_length = eval(iniconfig["experiment"]["flop_schedule_length"])

distill_temperature = eval(iniconfig["experiment"]["distill_temperature"])

beta1 = eval(iniconfig["experiment"]["beta1"])
beta2 = eval(iniconfig["experiment"]["beta2"])
weight_decay = eval(iniconfig["experiment"]["weight_decay"])
flop_loss_weight = eval(iniconfig["experiment"]["flop_loss_weight"])

wandb_project = iniconfig["logging"]["wandb_project"]


learning_rate = (0.3**0.5) * ((1.0 / est_p_count) ** 0.5 * (batch_dim / 16) ** 0.5)

optimizer_config = {
    "type": "Adam",
    "config": {"beta1": beta1, "beta2": beta2, "learning_rate": learning_rate},
}
import json
def json_serializable(obj):
    try:
        json.dumps(obj)
        return True
    except Exception:
        return False

run_config = {
    "dataset_name": dataset_name,
    "optimizer": optimizer_config,
    "train_schedule_length": train_schedule_length,
    "flop_schedule_length": flop_schedule_length,
    "batch_dim": batch_dim,
    "grad_accums": grad_accums,
    "grad_batch_dim": grad_batch_dim,
    "max_train_seq_len": max_train_seq_len,
    "distill_temperature": distill_temperature,
    "weight_decay": weight_decay,
    "cwic_config": dict(list(filter(json_serializable, vars(config).items())))
}


# Create optimizer
optimizer = nnx.Optimizer(
    student_model,
    optax.MultiSteps(
    optax.chain(
        clip_by_global_norm_quantile(history_length=100, top_k=10),
        optax.scale_by_adam(beta1, beta2),
        optax.add_decayed_weights(weight_decay),
        optax.scale_by_schedule(optax.schedules.warmup_constant_schedule(0.0, 1.0, 400)),
        optax.scale(-learning_rate),
    ),
        every_k_schedule=grad_accums,
    ),  # pyright: ignore[reportArgumentType]
    wrt=nnx.Param,
)

# # Create checkpoint manager for restore model
options = ocp.CheckpointManagerOptions(max_to_keep=4 if local_ckpts else 32)
# Create checkpoint manager for new model
model_date_time = str(datetime.datetime.now())
import os

save_path = os.path.join(os.getcwd(), args.save_checkpoints_to) if local_ckpts else args.save_checkpoints_to
save_path = f"{save_path}/{model_date_time.replace(" ", "_")}"
mngr = (
    ocp.CheckpointManager(
        save_path,
        options=options,
        item_names=("model", "optimizer", "data_state", "run_config", "train_run_state"),
    )
    if save_ckpts
    else None
)
data_state = None
train_run_state = {
    "batch_index": 0,
    "seen_tokens/combined": 0,
    "seen_tokens/assistant": 0,
    "seen_tokens/other": 0,
    "wandb_run_id": None,
}

# Restoring from an old path
# if args.restore_path != "NO_RUN":
# (nnx_model, optimizer, data_state, run_config, train_run_state) = restore_nnx_from_ckpt(
#     nnx_model, optimizer, args.restore_path, q=q
# )

# print("Restoring from checkpoint...")

# graph_def, state = nnx.split(nnx_model)
# # replace RNG key by a dummy to allow checkpoint restoration
# rngs_key = jax.tree.map(jax.random.key_data, state.filter(nnx.RngKey))
# merge_state(state, rngs_key)

# opt_def, opt_state = nnx.split(optimizer.opt_state)

# resume_step = restore_mngr.latest_step()
# restored = restore_mngr.restore(
#     resume_step,
#     args=ocp.args.Composite(
#         model=ocp.args.StandardRestore(nnx.eval_shape(lambda: state)),
#         optimizer=ocp.args.StandardRestore(nnx.eval_shape(lambda: opt_state)),
#         data_state=ocp.args.JsonRestore(),
#         run_config=ocp.args.JsonRestore(),
#         train_run_state=ocp.args.JsonRestore(),
#     ),
# )
# nnx_model = nnx.merge(graph_def, restored["model"])

# del rngs_key
# del graph_def
# del state

# # restore RNG key
# rngs_key = jax.tree.map(jax.random.wrap_key_data, restored["model"].filter(nnx.RngKey))
# merge_state(restored["model"], rngs_key)
# optimizer.optimizer = nnx.merge(opt_def, restored["optimizer"])

# del opt_def
# del opt_state
# del rngs_key

# data_state = restored["data_state"]
# run_config = restored["run_config"]
# train_run_state = restored["train_run_state"]

# del restored
# gc.collect()
# print("Restored.")


def data_generator(key, dataset_name, validation=False, state_dict=None):
    ds = datasets.load_dataset(
        dataset_name,
        split="train",
        streaming=True,
    )
    assert isinstance(ds, datasets.IterableDataset)

    ds = datasets.concatenate_datasets([ds] * epochs)
    if state_dict is not None:
        ds.load_state_dict(state_dict)
    dsi = iter(ds)

    def s():
        Q = next(dsi)
        return Q["input_ids.npy"], Q["segment_ids.npy"], Q["pad_mask.npy"], Q["gen_mask.npy"]

    while True:
        a, b, c, d = [], [], [], []
        for _ in range(batch_dim):
            a_, b_, c_, d_ = s()
            a.append(np.array(a_)[:max_train_seq_len])
            b.append(np.array(b_)[:max_train_seq_len])
            c.append(np.array(c_)[:max_train_seq_len])
            d.append(np.array(d_)[:max_train_seq_len])
        a = np.stack([a_ for a_ in a])
        b = np.stack([b_ for b_ in b])
        c = np.stack([c_ for c_ in c])
        d = np.stack([d_ for d_ in d])
        batch = (a, b, c, d)

        yield batch, ds.state_dict()
        del batch


def js_divergence(x, y):
    return (
        optax.losses.kl_divergence_with_log_targets(x, y)
        + optax.losses.kl_divergence_with_log_targets(y, x)
    ) / 2


def loss_fn(
    student_model: NNXRefractionModule, t_logits: Array, batch, flop_progress, key
):
    input_ids, segment_ids, pad_mask, gen_mask = batch

    # get model output
    s_logits, aux = student_model(
        input_ids,
        segment_ids,
        pad_mask,
        jnp.arange(input_ids.shape[-1])[None],
        key,
        gen_mask=gen_mask,
    )

    # kl loss
    teacher_logprobs = jax.nn.log_softmax(t_logits / distill_temperature, -1)
    student_logprobs = jax.nn.log_softmax(s_logits / distill_temperature, -1)

    kl_loss = js_divergence(teacher_logprobs, student_logprobs) * distill_temperature**2
    kl_loss = jnp.mean(kl_loss, where=(pad_mask))

    kl_raw = optax.losses.kl_divergence_with_log_targets(
        jax.nn.log_softmax(s_logits), jax.nn.log_softmax(t_logits)
    )
    kl_raw = jnp.mean(kl_raw, where=(pad_mask & gen_mask))

    # adjust losses
    num_layers = og_model_config.num_hidden_layers
    num_trees = 1 + (num_layers * 3)

    aux["in_bandwidth"] = aux.pop("in_bandwidth") / num_trees

    flop_og = aux[LossKeys.FLOPS_CWIC].mean(where=pad_mask) / aux[LossKeys.FLOPS_BASE].mean()
    flop_ratio = 1 / flop_og

    targ_glop = flop_progress * (flop_ratio_max - 1.0) + 1.0
    flop_loss = (targ_glop - flop_ratio).clip(0) ** 2

    loss = jnp.mean(kl_loss) + flop_loss_weight * jnp.mean(flop_loss)
    gs = pad_mask.astype(jnp.float32).mean()
    loss = loss + (gs - 1.0) * (loss - jax.lax.stop_gradient(loss))

    aux = {
        **aux,
        "kd_loss/combined": loss,
        LossKeys.LOGITS_TWOWAY: kl_loss,
        LossKeys.LOGITS_TEMP_1: kl_raw,
        "flop_ratios/combined": flop_ratio,
        "flop_loss/combined": flop_loss,
    }

    return loss, aux


def get_progress(x, l):
    return jnp.min(jnp.array([1.0, x / l])).mean()


@partial(nnx.jit)
def train_step_t_logits(
    teacher_model: NNXRefractionModule,
    batch,
    key,
):

    input_ids, segment_ids, pad_mask, gen_mask = batch
    teacher_model.eval()
    t_logits, _ = teacher_model(
        input_ids,
        segment_ids,
        pad_mask,
        jnp.arange(input_ids.shape[-1])[None],
        key,
        gen_mask=gen_mask,
    )
    return t_logits


@partial(nnx.jit, static_argnames=["no_opt", "init_mode"],donate_argnums=(3,))
def train_step(
    teacher_model: NNXRefractionModule,
    student_model: NNXRefractionModule,
    optimizer: nnx.Optimizer,
    batch,
    step,
    key,
    no_opt=False,
    init_mode=False,
):
    flop_progress = get_progress(step, flop_schedule_length)
    student_model.set_init_mode(init_mode)

    input_ids, segment_ids, pad_mask, gen_mask = batch
    teacher_model.eval()
    t_logits, _ = teacher_model(
        input_ids,
        segment_ids,
        pad_mask,
        jnp.arange(input_ids.shape[-1])[None],
        key,
        gen_mask=gen_mask,
    )
    t_logits=jax.lax.stop_gradient(t_logits)

    if no_opt:
        student_model.eval()
        loss, aux = loss_fn(
            student_model,
            t_logits,
            batch,
            flop_progress,
            key,
        )

    else:
        student_model.train()
        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, aux), grads = grad_fn(
            student_model,
            t_logits,
            batch,
            flop_progress,
            key,
        )

        grads = jax.tree.map(jnp.nan_to_num, grads)
        optimizer.update(student_model, grads)

    return loss, aux


run_config["checkpoint_save_path"] = save_path
run_tags = [] if args.tags is None else args.tags.split(",")
run = wandb.init(
    project=wandb_project,
    name=args.name,
    config=run_config,
    resume="allow",
    # id=train_run_state["wandb_run_id"],
    tags=[checkpoint, dataset_name] + run_tags,
    notes=args.notes,
)
train_run_state["wandb_run_id"] = run.id

tkey = jax.random.key(7)
train_random_key, data_key = jax.random.split(tkey)
last_diff = 0.0

initted = True  # train_run_state["batch_index"] > 0

with mesh:
    while True:

        prog = tqdm.tqdm(
            data_generator(data_key, dataset_name, state_dict=data_state),
            initial=train_run_state["batch_index"],
        )

        val_batch = None
        for batch, data_state in prog:
            if val_batch is None:
                val_batch = jax.tree.map(lambda x: x[:grad_batch_dim], batch)

            loss_disc = None
            aux_discrete = None
            data_sharding = NamedSharding(mesh, PartitionSpec("dp", None))

            loss, aux = -1.0, {}

            for no_opt in [False]:
                aves = {}
                pad_sum = 0
                gen_sum = 0
                for rp in range(grad_accums):
                    batch1 = jax.tree.map(
                        lambda x: x[rp * grad_batch_dim : (rp + 1) * grad_batch_dim], batch
                    )
                    batch2 = jax.tree.map(
                        lambda x: jax.device_put(jnp.array(x), data_sharding), batch1
                    )
                    # print("IN BATCH SHARDING")
                    # jax.debug.visualize_array_sharding(batch2[0])
                    

                    # print("OUT LOGIT SHARDING")
                    # jax.debug.visualize_array_sharding(t_logits[...,0])
                    res = train_step(
                        teacher_model,
                        student_model,
                        optimizer,
                        batch2,
                        train_run_state["batch_index"],
                        key=train_random_key,
                        no_opt=no_opt,
                        init_mode=not initted,
                    )
                    input_ids, segment_ids, pad_mask, gen_mask = batch1
                    addpad = int(pad_mask.sum())
                    pad_sum += addpad

                    addgen = int((pad_mask & gen_mask).sum())
                    gen_sum += addgen
                    initted = True
                    if rp == 0:
                        aves = dict(res[1])
                        aves["flop_ratios/combinedi"] = 1 / aves["flop_ratios/combined"] * addpad
                        aves[LossKeys.LOGITS_TEMP_1] = float(aves[LossKeys.LOGITS_TEMP_1]) * addgen
                        aves[LossKeys.LOGITS_TWOWAY] = float(aves[LossKeys.LOGITS_TWOWAY]) * addpad
                        aves["kd_loss/combined"] = float(aves["kd_loss/combined"]) * addpad
                    else:
                        aux = res[1]
                        aves["flop_ratios/combinedi"] += 1 / aux["flop_ratios/combined"] * addpad
                        aves[LossKeys.LOGITS_TEMP_1] += float(aux[LossKeys.LOGITS_TEMP_1]) * addgen
                        aves[LossKeys.LOGITS_TWOWAY] += float(aux[LossKeys.LOGITS_TWOWAY]) * addpad
                        aves["kd_loss/combined"] += float(aux["kd_loss/combined"]) * addpad
                    if rp == grad_accums - 1:
                        loss, aux = res
                        aux["flop_ratios/combined"] = 1 / (aves["flop_ratios/combinedi"] / pad_sum)
                        aux[LossKeys.LOGITS_TEMP_1] = aves[LossKeys.LOGITS_TEMP_1] / gen_sum
                        aux[LossKeys.LOGITS_TWOWAY] = aves[LossKeys.LOGITS_TWOWAY] / pad_sum
                        aux["kd_loss/combined"] = aves["kd_loss/combined"] / pad_sum

            student_model.clamp_thresholds()
            metric_kl = aux[LossKeys.LOGITS_TEMP_1]

            train_run_state["seen_tokens/combined"] += int(jnp.sum(batch[2]))
            train_run_state["seen_tokens/assistant"] += int(jnp.sum(batch[3]))
            train_run_state["seen_tokens/other"] += int(jnp.sum(batch[2])) - int(jnp.sum(batch[3]))
            per_param_flop_ratios = {}
            per_param_flop_ratios["flop_ratios/lm_head"] = float(
                student_model.lm_head.flop_ratio_trace.value
            )
            for ii in range(config.num_hidden_layers):
                per_param_flop_ratios[f"flop_ratios/qkv_{ii:03d}"] = float(
                    student_model.h.blocks.attention.wqkv_i.flop_ratio_trace.value[ii]
                )
                per_param_flop_ratios[f"flop_ratios/o_{ii:03d}"] = float(
                    student_model.h.blocks.attention.wo_i.flop_ratio_trace.value[ii]
                )
                per_param_flop_ratios[f"flop_ratios/ffn_gate_{ii:03d}"] = float(
                    student_model.h.blocks.feed_forward.ffn_i.gate.flop_ratio_trace.value[ii]
                )
                per_param_flop_ratios[f"flop_ratios/ffn_{ii:03d}"] = float(
                    student_model.h.blocks.feed_forward.ffn_i.flop_ratio_trace.value[ii]
                )
            aux = {**aux, **per_param_flop_ratios}
            aux.pop(LossKeys.FLOPS_CWIC)
            aux.pop(LossKeys.FLOPS_BASE)

            if HIST_INTERVAL == True or (
                HIST_INTERVAL != False and train_run_state["batch_index"] % HIST_INTERVAL == 0
            ):
                val_batch = jax.tree.map(lambda x: jax.device_put(x, data_sharding), val_batch)
                _, aux_val = train_step(
                    teacher_model,
                    student_model,
                    optimizer,
                    val_batch,
                    train_run_state["batch_index"],
                    key=train_random_key,
                    no_opt=True,
                )
                for ii in [0, 12]:
                    if student_model.h.blocks.attention.wqkv_i.hist_trace is not None:
                        aux[f"trees/qkv_{ii:03d}"] = wandb.Image(
                            np.array(
                                student_model.h.blocks.attention.wqkv_i.hist_trace.value[ii] * 255
                            ),
                            "L",
                        )
                    if student_model.h.blocks.attention.wo_i.hist_trace is not None:
                        aux[f"trees/o_{ii:03d}"] = wandb.Image(
                            np.array(
                                student_model.h.blocks.attention.wo_i.hist_trace.value[ii] * 255
                            ),
                            "L",
                        )

                    if student_model.h.blocks.feed_forward.ffn_i.gate.hist_trace is not None:
                        aux[f"trees/ffn_gate_{ii:03d}"] = wandb.Image(
                            np.array(
                                student_model.h.blocks.feed_forward.ffn_i.gate.hist_trace.value[ii]
                                * 255
                            ),
                            "L",
                        )
                    if student_model.h.blocks.feed_forward.ffn_i.hist_trace is not None:
                        aux[f"trees/ffn_{ii:03d}"] = wandb.Image(
                            np.array(
                                student_model.h.blocks.feed_forward.ffn_i.hist_trace.value[ii]
                                * 255
                            ),
                            "L",
                        )
                if student_model.h.blocks.attention.wo_i.hist_trace is not None:
                    aux[f"trees/o_{6:03d}"] = wandb.Image(
                        np.array(student_model.h.blocks.attention.wo_i.hist_trace.value[6] * 255),
                        "L",
                    )
                if student_model.lm_head.hist_trace is not None:
                    aux["trees/lm_head"] = wandb.Image(
                        np.array(student_model.lm_head.hist_trace.value * 255), "L"
                    )

            prog.set_description_str(
                f"loss: {loss},temp1:{aux[LossKeys.LOGITS_TEMP_1]},fr:{aux["flop_ratios/combined"]}",
                refresh=False,
            )
            # prog.set_postfix(aux)

            train_run_state["batch_index"] += 1

            if train_run_state["batch_index"] % ((128 * 500) // batch_dim) == 0 and (mngr is not None):
                _, state = nnx.split(student_model)
                # The RNG key had to be convert to int to allow checkpoint saving
                rngs_key = jax.tree.map(jax.random.key_data, state.filter(nnx.RngKey))

                def merge_state(dst: nnx.State, src: nnx.State):
                    for k, v in src.items():
                        if isinstance(v, nnx.State):
                            merge_state(dst[k], v)
                        else:
                            dst[k] = v

                merge_state(state, rngs_key)

                _, opt_state = nnx.split(optimizer.opt_state)

                mngr.save(
                    train_run_state["batch_index"],
                    args=ocp.args.Composite(
                        model=ocp.args.StandardSave(  # pyright: ignore[reportCallIssue]
                            jax.tree.map(np.array, state)  # pyright: ignore[reportCallIssue]
                        ),
                        data_state=ocp.args.JsonSave(  # pyright: ignore[reportCallIssue]
                            data_state  # pyright: ignore[reportCallIssue]
                        ),
                        run_config=ocp.args.JsonSave(  # pyright: ignore[reportCallIssue]
                            run_config  # pyright: ignore[reportCallIssue]
                        ),
                        train_run_state=ocp.args.JsonSave(  # pyright: ignore[reportCallIssue]
                            train_run_state  # pyright: ignore[reportCallIssue]
                        ),
                        optimizer=ocp.args.StandardSave(  # pyright: ignore[reportCallIssue]
                            jax.tree.map(np.array, opt_state)  # pyright: ignore[reportCallIssue]
                        ),
                    ),
                )
                del state
                del opt_state
                del rngs_key
                print("Saved a checkpoint.")
            if train_run_state["batch_index"] % TEST_INTERVAL == 0:

                # print(do_inference([{"role": "user", "content": "Hi! What is Halftone?"}])[-1])
                import html

                new_infer_tok_count = 256

                table = wandb.Table(columns=["Model", "Generation"])
                aux_infer = None
                for li, name in enumerate(["teacher", "student"]):
                    if name == "teacher":
                        continue
                    token_id_list: List[int] = list(
                        tokenizer.apply_chat_template(
                            [
                                # {
                                #     "role": "system",
                                #     "content": "You are a friendly AI model that formats replies in markdown.",
                                # },
                                {
                                    "role": "user",
                                    "content": """Find the slope of a line through (5, 8) and (9, 9)""",
                                },
                            ],
                            add_generation_prompt=True,
                            tokenize=True,
                        )
                    )  # type: ignore

                    max_len = len(token_id_list) + new_infer_tok_count
                    for mm in range(len(token_id_list), max_len):
                        eos_pad_count = max_len - 1 - len(token_id_list)
                        toks = jnp.array([token_id_list + [eos_token_id] * eos_pad_count])
                        logits, aux_infer = infer_nnx_model(
                            [teacher_model, student_model][li],
                            toks,
                            jnp.zeros_like(toks),
                            jnp.ones_like(toks).astype(jnp.bool),
                            jnp.cumsum(jnp.ones_like(toks), axis=-1) - 1,
                            jax.random.key(0),
                        )
                        new_token = int(jnp.argmax(logits[:, :], axis=-1)[0][mm - 1])

                        token_id_list.append(new_token)
                    strr = tokenizer.decode(token_id_list)
                    if name == "student":
                        assert aux_infer is not None
                        colors = np.array(
                            (
                                (
                                    aux_infer[LossKeys.FLOPS_CWIC]
                                    / np.mean(aux_infer[LossKeys.FLOPS_CWIC])
                                ).ravel()
                            )
                        )
                        mc = np.array(
                            (
                                (
                                    aux_infer[LossKeys.FLOPS_CWIC] / aux_infer[LossKeys.FLOPS_BASE]
                                ).ravel()
                            )
                        ).tolist()
                        colors = np.log10(colors)
                        colors = colors - colors.mean()
                        colors = colors / (np.std(colors) + 0.0001) * 0.5
                        colors = np.argsort(np.argsort(colors)) / colors.shape[0]
                        colors = colors.tolist()

                        def get_color(c, a):
                            c = c + 0.5
                            return f"rgba(0,0,0,{a})"

                        aux[f"generation/rich"] = wandb.Html(
                            """<link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,100..900;1,14..32,100..900&display=swap" rel="stylesheet">"""
                            + "<pre>"
                            + "".join(
                                [
                                    # background:linear-gradient(90deg,{get_color(c,a1)},{get_color(c2,a2)});
                                    f'<span style="font-family: Inter;font-optical-sizing: auto;font-weight: {100+int(800*((c*a1+c2*a2*0.0001)/(a1+a2*0.0001))):3d};font-style: normal;">{html.escape(tokenizer.decode([m]))}</span>'
                                    for m, c, c2, a1, a2 in zip(
                                        token_id_list,
                                        [colors[0]] + colors,
                                        colors + [colors[-1]],
                                        [0.0] + [1.0] * (len(token_id_list) - 1),
                                        [1.0] * (len(token_id_list) - 1) + [0.0],
                                    )
                                ]
                            )
                            + "</pre>"
                        )

                        table.add_data(
                            name + "_token_decodes",
                            json.dumps([tokenizer.decode([m]) for m in token_id_list]),
                        )
                        table.add_data(name + "_flop_ratios", json.dumps(mc))
                    table.add_data(name, strr)
                    print(name, strr)
                if aux_infer is not None:
                    aux[f"generation/table"] = table
            if mngr is not None:
                mngr.wait_until_finished()
            aux = {
                **aux,
                "seen_tokens/assistant": train_run_state["seen_tokens/assistant"],
                "seen_tokens/other": train_run_state["seen_tokens/other"],
                "seen_tokens/combined": train_run_state["seen_tokens/combined"],
            }
            run.log(aux)

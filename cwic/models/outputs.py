from typing import Optional

from torch import Tensor
from dataclasses import dataclass

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast


@dataclass
class BaseModelOutputWithPastAndActiveParameters(BaseModelOutputWithPast):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Also contains information about active and dense parameters for each token in the batch.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        active_parameters (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            The number of active parameters for each token in the batch.
        dense_parameters (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            The number of dense parameters (as in, the total number of parameters that would be used without sparsity) for each token in the batch.

            Always the same for each token, and should not change during training.
    """

    active_parameters: Optional[Tensor] = None
    dense_parameters: Optional[Tensor] = None


@dataclass
class CausalLMOutputWithPastAndActiveParameters(CausalLMOutputWithPast):
    """
    Base class for causal language model (or autoregressive) outputs.

    Also contains information about active and dense parameters for each token in the batch.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        active_parameters (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            The number of active parameters for each token in the batch.
        dense_parameters (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            The number of dense parameters (as in, the total number of parameters that would be used without sparsity) for each token in the batch.

            Always the same for each token, and should not change during training.
    """

    active_parameters: Optional[Tensor] = None
    dense_parameters: Optional[Tensor] = None

import torch

from transformers import pipeline

from models.convert import llama_to_cwic
from models.configuration_cwic import CWICConfig
from models.modelling_cwic import CWICForCausalLM


def main():

    messages = [
        {
            "role": "system",
            "content": "You are a pirate chatbot who always responds in pirate speak!",
        },
        {"role": "user", "content": "Who are you?"},
    ]

    pipe = pipeline(
        "text-generation",
        model="unsloth/Llama-3.2-1B-Instruct",
        device="cpu",
        torch_dtype=torch.float32,
    )
    print("Pipe Loaded!")

    outputs = pipe(
        messages,
        max_new_tokens=16,
        do_sample=False,
    )
    og_outputs = outputs[0]["generated_text"][-1]["content"]
    print("\n === Reference Output ===")
    print(og_outputs)
    print("")

    cwic_model = llama_to_cwic(pipe.model, num_stripes=4, num_head_stripes=7)
    cwic_model.eval()
    print("CWIC Model Converted!")

    pipe.model = cwic_model

    outputs = pipe(
        messages,
        max_new_tokens=16,
        do_sample=False,
    )
    cwic_outputs = outputs[0]["generated_text"][-1]["content"]
    print("\n === CWIC Output ===")
    print(cwic_outputs)
    print("")

    print(f"Outputs match: {og_outputs == cwic_outputs}")


if __name__ == "__main__":
    main()

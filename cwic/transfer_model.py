
import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer

from models.modelling_cwic import CWICForCausalLM


def main(args):
    assert (args.download is None) != (args.upload is None), "Must provide exactly one of --download or --upload"

    if args.download is not None:

        model = AutoModelForCausalLM.from_pretrained(args.repo, device_map="cpu")
        tokenizer = AutoTokenizer.from_pretrained(args.repo)

        model.save_pretrained(args.download)
        tokenizer.save_pretrained(args.download)

        print(f"\nModel and tokenizer from {args.repo} saved to {args.download}\n")

    else:

        model = AutoModelForCausalLM.from_pretrained(args.upload, device_map="cpu")
        tokenizer = AutoTokenizer.from_pretrained(args.upload)

        model.push_to_hub(args.repo, private=True)
        tokenizer.push_to_hub(args.repo, private=True)

        print(f"\nModel and tokenizer from {args.upload} uploaded to {args.repo}\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--repo", type=str, required=True)
    parser.add_argument("--download", type=str, default=None)
    parser.add_argument("--upload", type=str, default=None)
    args = parser.parse_args()

    main(args)
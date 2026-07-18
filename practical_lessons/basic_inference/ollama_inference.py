import argparse
import sys

from ollama import Client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a basic Llama 3.2 inference through Ollama."
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="User prompt for the model.",
    )
    parser.add_argument(
        "--model",
        default="llama3.2:1b",
        help="Ollama model name. Default uses the lightest Llama 3.2 model.",
    )
    parser.add_argument(
        "--host",
        default="http://localhost:11434",
        help="Ollama server URL.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        client = Client(host=args.host)
        response = client.chat(
            model=args.model,
            messages=[{"role": "user", "content": args.prompt}],
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error during Ollama inference: {exc}", file=sys.stderr)
        return 1

    print(response["message"]["content"].strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import os
import sys

from dotenv import load_dotenv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a basic LLM inference using OpenAI or Anthropic."
    )
    parser.add_argument(
        "--provider",
        required=True,
        choices=["openai", "anthropic"],
        help="LLM provider to use.",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="User prompt for the model.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name. If omitted, a default model for the provider is used.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum output tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    return parser.parse_args()


def run_openai(prompt: str, model: str, max_tokens: int, temperature: float) -> str:
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY in environment variables.")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


def run_anthropic(prompt: str, model: str, max_tokens: int, temperature: float) -> str:
    from anthropic import Anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Missing ANTHROPIC_API_KEY in environment variables.")

    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )

    text_chunks = [block.text for block in response.content if block.type == "text"]
    return "".join(text_chunks)


def main() -> int:
    load_dotenv()
    args = parse_args()

    provider_defaults = {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-5-haiku-latest",
    }
    model = args.model or provider_defaults[args.provider]

    try:
        if args.provider == "openai":
            output = run_openai(
                prompt=args.prompt,
                model=model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
        else:
            output = run_anthropic(
                prompt=args.prompt,
                model=model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
    except Exception as exc:  # noqa: BLE001
        print(f"Error during inference: {exc}", file=sys.stderr)
        return 1

    print(output.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

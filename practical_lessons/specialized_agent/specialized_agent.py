import argparse
import os
import sys

import tiktoken
from dotenv import load_dotenv
from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a specialized chatbot with OpenAI-compatible API against local Ollama."
        )
    )
    parser.add_argument(
        "--personality",
        required=True,
        help="Agent personality and behavior instructions.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--output-tokens",
        type=int,
        default=256,
        help="Maximum output tokens per response.",
    )
    parser.add_argument(
        "--input-tokens",
        type=int,
        default=512,
        help="Maximum input tokens allowed per user message.",
    )
    parser.add_argument(
        "--language",
        default="english",
        help="Agent language for responses. Default is english.",
    )
    parser.add_argument(
        "--model",
        default="llama3.2:1b",
        help="Local Ollama model exposed through OpenAI-compatible API.",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434/v1",
        help="OpenAI-compatible base URL for local Ollama.",
    )
    return parser.parse_args()


def count_tokens(text: str, model: str) -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def build_system_prompt(personality: str, language: str) -> str:
    return (
        "You are a specialized assistant with the following personality: "
        f"{personality}. "
        f"Always answer in {language}. "
        "Be concise, accurate, and practical. Ask clarifying questions when needed."
    )


def main() -> int:
    load_dotenv()
    args = parse_args()

    api_key = os.getenv("OPENAI_API_KEY", "ollama")
    client = OpenAI(api_key=api_key, base_url=args.base_url)

    messages = [
        {
            "role": "system",
            "content": build_system_prompt(args.personality, args.language),
        }
    ]

    print("Specialized agent started. Press Ctrl+C to exit.")

    try:
        while True:
            user_text = input("You: ").strip()
            if not user_text:
                continue

            token_count = count_tokens(user_text, args.model)
            print(f"Input tokens for this message: {token_count}")
            if token_count > args.input_tokens:
                print(
                    "Message exceeds --input-tokens limit. "
                    "Please shorten your message and try again.",
                    file=sys.stderr,
                )
                continue

            messages.append({"role": "user", "content": user_text})

            response = client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.output_tokens,
            )

            assistant_text = response.choices[0].message.content or ""
            assistant_text = assistant_text.strip()
            print(f"Agent: {assistant_text}")
            messages.append({"role": "assistant", "content": assistant_text})
    except KeyboardInterrupt:
        print("\nChat ended by user (Ctrl+C).")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"Error during chat: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import base64
import sys
from pathlib import Path
from ollama import Client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OCR Agent script using Ollama vision/multimodal models."
    )
    parser.add_argument("--image-path", required=True, help="Path to the input image file.")
    parser.add_argument(
        "--model",
        required=True,
        help="Multimodal/vision model name to perform OCR (e.g., llama3.2-vision).",
    )
    parser.add_argument(
        "--host", default="http://localhost:11434", help="Ollama server URL."
    )
    return parser.parse_args()


def encode_image_to_base64(image_path: Path) -> str:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at: {image_path}")
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def main() -> int:
    args = parse_args()
    image_path = Path(args.image_path).expanduser().resolve()

    if not image_path.exists() or image_path.suffix.lower() not in [".png", ".jpg", ".jpeg", ".webp"]:
        print(
            "Error: Invalid or non-existent image path. Supported extensions: .png, .jpg, .jpeg, .webp",
            file=sys.stderr,
        )
        return 1

    try:
        print(f"Encoding image {image_path.name} to base64...")
        base64_image = encode_image_to_base64(image_path)
    except Exception as e:
        print(f"Error reading/encoding image: {e}", file=sys.stderr)
        return 1

    client = Client(host=args.host)

    system_prompt = (
        "You are an expert OCR engine. Analyze the provided image and extract all text "
        "exactly as it appears. Maintain structural layouts where possible (like lists or tables). "
        "Provide your output formatted in clean Markdown. Do not include any conversational filler "
        "or explanations, only return the extracted text in Markdown."
    )

    messages = [
        {
            "role": "user",
            "content": system_prompt,
            "images": [base64_image],
        }
    ]

    print(f"Sending request to model '{args.model}' on {args.host} for OCR extraction...")
    try:
        response = client.chat(model=args.model, messages=messages)
        extracted_text = (response.get("message", {}).get("content") or "").strip()

        if not extracted_text:
            print("Error: Model returned an empty response.", file=sys.stderr)
            return 1

        output_path = image_path.with_name(f"{image_path.stem}_ocr.md")
        print(f"Saving extracted Markdown content to: {output_path}")
        output_path.write_text(extracted_text + "\n", encoding="utf-8")
        
        print("\n--- Extracted Text Preview ---")
        print(extracted_text)
        print("------------------------------")
        
    except Exception as e:
        print(f"Error during OCR agent execution: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

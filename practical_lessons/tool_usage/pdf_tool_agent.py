import argparse
import json
import sys
from pathlib import Path
from typing import Any, Literal, Optional

import fitz
from ollama import Client
from pydantic import BaseModel, Field


class AgentResponse(BaseModel):
    action: Literal["extract_content", "extract_images", "answer", "done"] = Field(
        description="The action to take: 'extract_content' to get text, 'extract_images' to get images, 'answer' to update progress, or 'done' if the goal is fully achieved."
    )
    formatted_query: Optional[str] = Field(
        default=None,
        description="Search query to extract relevant content. Required if action is 'extract_content'."
    )
    final_answer: Optional[str] = Field(
        default=None,
        description="The final answer or progress description. Required if action is 'answer' or 'done'."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Agentic Ollama script with PDF tools for content extraction and image extraction."
        )
    )
    parser.add_argument("--model", default="llama3.2:1b", help="Main agent model name.")
    parser.add_argument(
        "--selector-model",
        default="llama3.2:1b",
        help="Model used by content selection tool.",
    )
    parser.add_argument("--pdf-path", required=True, help="Absolute or relative path to input PDF.")
    parser.add_argument("--query", required=True, help="User objective/query for the agent.")
    parser.add_argument(
        "--host", default="http://localhost:11434", help="Ollama server URL."
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=8,
        help="Maximum agent loop steps before stopping.",
    )
    return parser.parse_args()


def format_pdf_text(pdf_path: Path) -> str:
    pages: list[str] = []
    with fitz.open(pdf_path) as doc:
        for page_index, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if not text:
                continue
            pages.append(f"## Page {page_index}\n{text}")
    return "\n\n".join(pages)


def extract_content_to_markdown(
    pdf_path: Path,
    formatted_user_query: str,
    client: Client,
    selector_model: str,
) -> Path:
    pdf_text = format_pdf_text(pdf_path)
    if not pdf_text:
        raise ValueError("No text content found in PDF.")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a PDF content selector. Follow the user query exactly and "
                "return only the best matching parts of the PDF content in Markdown. "
                "When useful, include the page heading references from the provided text."
            ),
        },
        {
            "role": "user",
            "content": (
                f"User query:\n{formatted_user_query}\n\n"
                "PDF content (pre-formatted):\n"
                f"{pdf_text}"
            ),
        },
    ]

    response = client.chat(model=selector_model, messages=messages)
    selected_text = (response.get("message", {}).get("content") or "").strip()
    if not selected_text:
        raise ValueError("Selector model returned empty content.")

    output_path = pdf_path.with_name(f"{pdf_path.stem}_selected_content.md")
    output_path.write_text(selected_text + "\n", encoding="utf-8")
    return output_path


def extract_images_from_pdf(pdf_path: Path) -> list[Path]:
    saved: list[Path] = []
    with fitz.open(pdf_path) as doc:
        for page_idx, page in enumerate(doc, start=1):
            for image_idx, image_data in enumerate(page.get_images(full=True), start=1):
                xref = image_data[0]
                pix = fitz.Pixmap(doc, xref)
                if pix.n > 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)

                output = pdf_path.with_name(
                    f"{pdf_path.stem}_page_{page_idx}_image_{image_idx}.png"
                )
                pix.save(output)
                pix = None
                saved.append(output)
    return saved


def build_agent_system_prompt() -> str:
    return (
        "You are an agentic assistant that can use tools to solve the user goal. "
        "Available actions: extract_content, extract_images, answer, done. "
        "Always respond with strict JSON: "
        '{"action":"...","formatted_query":"...","final_answer":"..."}. '
        "Rules: "
        "1) If content extraction is needed, choose extract_content and provide formatted_query. "
        "2) If image extraction is needed, choose extract_images. "
        "3) Use answer for progress updates. "
        "4) Use done only when the goal is reached."
    )


def parse_model_json(content: str) -> dict[str, Any]:
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("Model response does not contain JSON object.")
    return json.loads(content[start : end + 1])


def main() -> int:
    args = parse_args()
    pdf_path = Path(args.pdf_path).expanduser().resolve()

    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        print("Invalid --pdf-path. It must point to an existing .pdf file.", file=sys.stderr)
        return 1

    client = Client(host=args.host)
    execution_log: list[str] = []
    last_response_text = "No model response yet."

    messages: list[dict[str, str]] = [
        {"role": "system", "content": build_agent_system_prompt()},
        {
            "role": "user",
            "content": (
                f"Goal: {args.query}\n"
                f"PDF path: {pdf_path}\n"
                "Decide the next action using JSON only."
            ),
        },
    ]

    for step in range(1, args.max_steps + 1):
        print(f"\n=== Agent Step {step} ===")
        print(f"Last model response: {last_response_text}\n")

        try:
            response = client.chat(
                model=args.model,
                messages=messages,
                format=AgentResponse.model_json_schema(),
            )
            model_text = (response.get("message", {}).get("content") or "").strip()
            if not model_text:
                raise ValueError("Main model returned empty response.")

            last_response_text = model_text
            print(f"Model output: {model_text}\n")

            action_payload = parse_model_json(model_text)
            validated_payload = AgentResponse.model_validate(action_payload)
            action = validated_payload.action.strip().lower()

            if action == "extract_content":
                formatted_query = validated_payload.formatted_query or args.query
                output_path = extract_content_to_markdown(
                    pdf_path=pdf_path,
                    formatted_user_query=formatted_query,
                    client=client,
                    selector_model=args.selector_model,
                )
                tool_result = f"extract_content executed. Markdown saved at: {output_path}"
            elif action == "extract_images":
                images = extract_images_from_pdf(pdf_path)
                if images:
                    tool_result = (
                        "extract_images executed. PNG files saved:\n"
                        + "\n".join(str(path) for path in images)
                    )
                else:
                    tool_result = (
                        "extract_images executed. No images were found in this PDF."
                    )
            elif action == "answer":
                tool_result = validated_payload.final_answer or "Progress update provided without final answer."
            elif action == "done":
                final_answer = validated_payload.final_answer or "Goal reached."
                print(f"Final answer: {final_answer}")
                print("\nFunction execution log:")
                for item in execution_log:
                    print(f"- {item}")
                return 0
            else:
                tool_result = (
                    "Unknown action returned by model. Please return a valid action."
                )

            execution_log.append(tool_result)
            messages.append({"role": "assistant", "content": model_text})
            messages.append({"role": "user", "content": f"Tool result:\n{tool_result}"})
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            return 0
        except Exception as exc:  # noqa: BLE001
            err = f"Error in agent loop: {exc}"
            execution_log.append(err)
            messages.append({"role": "user", "content": err})
            print(err, file=sys.stderr)

    print("Max steps reached before goal confirmation.", file=sys.stderr)
    print("\nFunction execution log:")
    for item in execution_log:
        print(f"- {item}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

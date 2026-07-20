import argparse
import sys
from pathlib import Path
import fitz
from mcp.server.fastmcp import FastMCP

# Initialize the MCP Server
mcp = FastMCP("Markdown to PDF Server")


@mcp.tool()
def convert_markdown_to_pdf(markdown_content: str, output_path: str) -> str:
    """
    Convert markdown text content to a PDF file and save it to the specified path.

    Args:
        markdown_content: The markdown text content to convert.
        output_path: The absolute path where the PDF file should be saved.
    """
    try:
        out_path = Path(output_path).resolve()
        # Create parent directories if they don't exist
        out_path.parent.mkdir(parents=True, exist_ok=True)

        doc = fitz.open()
        page = doc.new_page()
        
        y = 50
        lines = markdown_content.splitlines()
        for line in lines:
            # Handle page breaks if content overflows
            if y > 750:
                page = doc.new_page()
                y = 50

            if line.startswith("# "):
                fontsize = 20
                text = line[2:]
                fontname = "helv-bold"
            elif line.startswith("## "):
                fontsize = 16
                text = line[3:]
                fontname = "helv-bold"
            elif line.startswith("### "):
                fontsize = 14
                text = line[4:]
                fontname = "helv-bold"
            else:
                fontsize = 11
                text = line
                fontname = "helv"
                
            if not text.strip():
                y += 10
                continue
                
            rect = fitz.Rect(50, y, 550, y + 100)
            page.insert_textbox(rect, text, fontsize=fontsize, fontname=fontname)
            
            # Simple text wrap space estimation
            line_count = max(1, len(text) // 60)
            y += fontsize + line_count * 14 + 6

        doc.save(str(out_path))
        doc.close()
        return f"Successfully saved PDF to {out_path}"
    except Exception as e:
        return f"Error converting markdown to PDF: {str(e)}"


def main() -> int:
    parser = argparse.ArgumentParser(description="MCP server for converting Markdown to PDF.")
    parser.add_argument("--run-mcp", action="store_true", help="Run as stdio MCP server.")
    parser.add_argument("--model", default="llama3.2:1b", help="Ollama model to use for the demo.")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama server URL.")
    args = parser.parse_args()

    # If --run-mcp flag is passed, run the MCP server using stdio transport
    if args.run-mcp:
        mcp.run()
        return 0

    # Otherwise, run a demo showcasing tool invocation by a model call
    from ollama import Client

    # Define the tool structure expected by Ollama
    pdf_tool = {
        "type": "function",
        "function": {
            "name": "convert_markdown_to_pdf",
            "description": "Convert markdown text content to a PDF file and save it to the specified path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "markdown_content": {
                        "type": "string",
                        "description": "The markdown text content to convert."
                    },
                    "output_path": {
                        "type": "string",
                        "description": "The absolute path where the PDF file should be saved."
                    }
                },
                "required": ["markdown_content", "output_path"]
            }
        }
    }

    client = Client(host=args.host)
    messages = [
        {
            "role": "user",
            "content": "Please convert this markdown content: '# Hello PDF\\nThis is generated from an LLM call.' and save it as a PDF at './output_mcp.pdf'"
        }
    ]

    print("Sending prompt to model to trigger MCP tool...")
    try:
        response = client.chat(
            model=args.model,
            messages=messages,
            tools=[pdf_tool]
        )

        message = response.get("message", {})
        print(f"\nModel Response: {message}")

        if "tool_calls" in message:
            for tool_call in message["tool_calls"]:
                func_name = tool_call["function"]["name"]
                arguments = tool_call["function"]["arguments"]
                print(f"\nModel triggered tool: {func_name}")
                print(f"Arguments: {arguments}")
                
                if func_name == "convert_markdown_to_pdf":
                    res = convert_markdown_to_pdf(
                        markdown_content=arguments.get("markdown_content"),
                        output_path=arguments.get("output_path")
                    )
                    print(f"Tool Execution Result: {res}")
        else:
            print("\nModel did not call the tool. Output content:")
            print(message.get("content"))
    except Exception as e:
        print(f"\nError running model call demo: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

from typing import Any


def extract_response_text(response: Any) -> str:
    content = (
        response
        if isinstance(response, str)
        else getattr(response, "content", None)
    )

    if isinstance(content, str) and content.strip():
        return content.strip()

    if isinstance(content, list):
        text_blocks = []
        for block in content:
            if isinstance(block, str):
                text_blocks.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                text_blocks.append(block["text"])
        text = "".join(text_blocks).strip()
        if text:
            return text

    raise RuntimeError("chat model returned an empty or unsupported response")

import re
import unicodedata


class TextNormalizer:
    _horizontal_whitespace = re.compile(r"[^\S\n]+")
    _excess_blank_lines = re.compile(r"\n{3,}")

    def normalize(self, content: str) -> str:
        normalized = unicodedata.normalize("NFC", content)
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
        normalized = "\n".join(
            self._horizontal_whitespace.sub(" ", line).strip()
            for line in normalized.split("\n")
        )
        normalized = self._excess_blank_lines.sub("\n\n", normalized)
        return normalized.strip()

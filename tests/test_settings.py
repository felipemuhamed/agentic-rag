import pytest
from pydantic import ValidationError

from app.settings import Settings


def test_max_rag_attempts_must_be_at_least_one() -> None:
    with pytest.raises(ValidationError):
        Settings(max_rag_attempts=0, _env_file=None)

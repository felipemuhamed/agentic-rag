import pytest
from pydantic import ValidationError

from app.settings import Settings


def test_max_rag_attempts_must_be_at_least_one() -> None:
    with pytest.raises(ValidationError):
        Settings(max_rag_attempts=0, _env_file=None)


@pytest.mark.parametrize("max_results", [0, 21])
def test_web_search_result_limit_is_validated(max_results: int) -> None:
    with pytest.raises(ValidationError):
        Settings(web_search_max_results=max_results, _env_file=None)


def test_web_search_timeout_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Settings(web_search_timeout_seconds=0, _env_file=None)

import pytest

from app.runtime import RagRuntime
from app.settings import Settings


def make_config(**overrides: object) -> Settings:
    return Settings(
        app_environment="test",
        visual_catalog_enabled=False,
        _env_file=None,
        **overrides,
    )


def test_warmup_runtime_is_not_ready_until_every_model_is_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.runtime.create_image_analyzer",
        lambda config, **options: object(),
    )
    runtime = RagRuntime(
        make_config(warmup_on_startup=True),
        graph_builder=lambda config, checkpointer: object(),
    )

    # An orchestrator must not route traffic here yet: the first stream would
    # spend its whole deadline loading weights.
    assert runtime.ready is False

    runtime.warmup()

    assert runtime.ready is True


def test_failed_warmup_keeps_readiness_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(config: Settings, **options: object) -> object:
        raise RuntimeError("modelo indisponível")

    monkeypatch.setattr("app.runtime.create_image_analyzer", explode)
    runtime = RagRuntime(
        make_config(warmup_on_startup=True),
        graph_builder=lambda config, checkpointer: object(),
    )

    with pytest.raises(RuntimeError):
        runtime.warmup()

    assert runtime.ready is False
    assert runtime.last_error == "RuntimeError"


def test_lazy_runtime_reports_ready_without_warmup() -> None:
    runtime = RagRuntime(make_config(warmup_on_startup=False))

    assert runtime.ready is True


def test_close_reopens_the_warmup_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.runtime.create_image_analyzer",
        lambda config, **options: object(),
    )
    runtime = RagRuntime(
        make_config(warmup_on_startup=True),
        graph_builder=lambda config, checkpointer: object(),
    )
    runtime.warmup()

    runtime.close()

    # The models were released, so readiness has to close with them.
    assert runtime.ready is False

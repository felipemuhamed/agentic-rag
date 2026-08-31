from contextlib import contextmanager

import pytest

from app.observability.tracing import (
    LangfuseTracing,
    NullTracing,
    TracingComponents,
    create_tracing,
)
from app.settings import Settings


class FakeSpan:
    def __init__(self, recorder: list[str], name: str):
        self._recorder = recorder
        self._name = name

    def __enter__(self) -> "FakeSpan":
        self._recorder.append(f"enter:{self._name}")
        return self

    def __exit__(self, *exc_info: object) -> bool:
        self._recorder.append(f"exit:{self._name}")
        return False


class FakeClient:
    def __init__(self, recorder: list[str] | None = None, **credentials: object):
        self.recorder = recorder if recorder is not None else []
        self.credentials = credentials
        self.shutdown_calls = 0

    def start_as_current_observation(self, *, as_type: str, name: str) -> FakeSpan:
        self.recorder.append(f"observation:{as_type}:{name}")
        return FakeSpan(self.recorder, "observation")

    def shutdown(self) -> None:
        self.shutdown_calls += 1


def make_tracing(recorder: list[str]) -> tuple[LangfuseTracing, FakeClient]:
    client = FakeClient(recorder)
    handler = object()

    @contextmanager
    def propagate_attributes(**attributes: object):
        recorder.append(f"propagate:{sorted(attributes)}")
        yield

    return (
        LangfuseTracing(
            client=client,
            handler=handler,
            propagate_attributes=propagate_attributes,
        ),
        client,
    )


def make_config(**overrides: object) -> Settings:
    return Settings(app_environment="test", _env_file=None, **overrides)


def test_tracing_is_off_by_default() -> None:
    tracing = create_tracing(make_config())

    assert isinstance(tracing, NullTracing)
    assert tracing.callbacks() == []


def test_null_tracing_context_is_transparent() -> None:
    tracing = NullTracing()

    with tracing.conversation(session_id="s", tags=["t"]):
        result = "resposta"

    assert result == "resposta"


def test_enabled_without_credentials_degrades_instead_of_raising() -> None:
    # A misconfigured collector must not stop the service from answering.
    tracing = create_tracing(make_config(langfuse_enabled=True))

    assert isinstance(tracing, NullTracing)


def test_missing_package_degrades_instead_of_raising() -> None:
    def loader() -> TracingComponents:
        raise ImportError("no module named langfuse")

    tracing = create_tracing(
        make_config(
            langfuse_enabled=True,
            langfuse_public_key="lf_pk_test",
            langfuse_secret_key="lf_sk_test",
        ),
        loader=loader,
    )

    assert isinstance(tracing, NullTracing)


def test_client_failure_degrades_instead_of_raising() -> None:
    def loader() -> TracingComponents:
        def explode(**credentials: object) -> object:
            raise RuntimeError("unreachable")

        return TracingComponents(
            client_factory=explode,
            handler_factory=object,
            propagate_attributes=lambda **attributes: None,
        )

    tracing = create_tracing(
        make_config(
            langfuse_enabled=True,
            langfuse_public_key="lf_pk_test",
            langfuse_secret_key="lf_sk_test",
        ),
        loader=loader,
    )

    assert isinstance(tracing, NullTracing)


def test_enabled_tracing_passes_configured_credentials_to_the_client() -> None:
    created: list[FakeClient] = []

    def loader() -> TracingComponents:
        def build(**credentials: object) -> FakeClient:
            client = FakeClient(**credentials)
            created.append(client)
            return client

        return TracingComponents(
            client_factory=build,
            handler_factory=lambda: "handler",
            propagate_attributes=lambda **attributes: None,
        )

    tracing = create_tracing(
        make_config(
            langfuse_enabled=True,
            langfuse_public_key="lf_pk_test",
            langfuse_secret_key="lf_sk_test",
            langfuse_base_url="http://langfuse-web:3000",
        ),
        loader=loader,
    )

    # The host is passed explicitly rather than read from the environment, so
    # the deployment does not depend on the SDK's env var naming.
    assert created[0].credentials == {
        "public_key": "lf_pk_test",
        "secret_key": "lf_sk_test",
        "base_url": "http://langfuse-web:3000",
    }
    assert tracing.callbacks() == ["handler"]


def test_conversation_opens_a_root_span_and_propagates_the_session() -> None:
    recorder: list[str] = []
    tracing, _ = make_tracing(recorder)

    with tracing.conversation(session_id="thread-abc", tags=["agentic-rag"]):
        recorder.append("trabalho")

    assert recorder == [
        "observation:span:chat-stream",
        "enter:observation",
        "propagate:['session_id', 'tags']",
        "trabalho",
        "exit:observation",
    ]


def test_failure_to_open_a_trace_still_runs_the_request() -> None:
    class BrokenClient(FakeClient):
        def start_as_current_observation(self, **options: object):
            raise RuntimeError("collector unreachable")

    tracing = LangfuseTracing(
        client=BrokenClient(),
        handler=object(),
        propagate_attributes=lambda **attributes: None,
    )

    executed = False
    with tracing.conversation(session_id="s", tags=["t"]):
        executed = True

    assert executed is True


def test_errors_inside_the_traced_body_still_propagate() -> None:
    tracing, _ = make_tracing([])

    # Guarding the span must not turn a real failure into a silent success.
    with pytest.raises(ValueError):
        with tracing.conversation(session_id="s", tags=["t"]):
            raise ValueError("falha real")


def test_shutdown_drains_the_client() -> None:
    tracing, client = make_tracing([])

    tracing.shutdown()

    assert client.shutdown_calls == 1


def test_shutdown_failure_is_swallowed() -> None:
    class BrokenClient(FakeClient):
        def shutdown(self) -> None:
            raise RuntimeError("already closed")

    tracing = LangfuseTracing(
        client=BrokenClient(),
        handler=object(),
        propagate_attributes=lambda **attributes: None,
    )

    tracing.shutdown()

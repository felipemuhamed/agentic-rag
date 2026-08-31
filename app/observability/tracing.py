"""Optional Langfuse tracing for the LLM calls in the pipeline.

Observability must never be able to take the product down: a tracing backend is
a nice-to-have, and an answer that fails because the trace collector is
unreachable is a worse outcome than an answer nobody can inspect. So every
failure path here degrades to a no-op and says so in the log, rather than
raising into a request.

The client is constructed explicitly from this project's settings instead of
letting the SDK read the environment. The SDK renamed its host variable between
major versions, and passing the value directly means the deployment does not
depend on which name the installed version happens to honour.
"""

import logging
from collections.abc import Callable, Iterator, Sequence
from contextlib import (
    AbstractContextManager,
    ExitStack,
    contextmanager,
    nullcontext,
)
from typing import Any, NamedTuple, Protocol

from app.settings import Settings


logger = logging.getLogger(__name__)

ROOT_OBSERVATION_NAME = "chat-stream"


class TracingComponents(NamedTuple):
    """The pieces of the Langfuse SDK this module needs.

    Resolving them through a loader keeps `create_tracing` testable without the
    package installed, and keeps the SDK import in exactly one place.
    """

    client_factory: Callable[..., Any]
    handler_factory: Callable[[], Any]
    propagate_attributes: Callable[..., Any]


def import_langfuse() -> TracingComponents:
    from langfuse import Langfuse, propagate_attributes
    from langfuse.langchain import CallbackHandler

    return TracingComponents(
        client_factory=Langfuse,
        handler_factory=CallbackHandler,
        propagate_attributes=propagate_attributes,
    )


class Tracing(Protocol):
    def callbacks(self) -> list[Any]:
        """Return LangChain callbacks to attach to a runnable invocation."""
        ...

    def conversation(
        self,
        *,
        session_id: str,
        tags: Sequence[str],
    ) -> AbstractContextManager[None]:
        """Group everything done for one request into a single trace."""
        ...

    def shutdown(self) -> None: ...


class NullTracing:
    def callbacks(self) -> list[Any]:
        return []

    def conversation(
        self,
        *,
        session_id: str,
        tags: Sequence[str],
    ) -> AbstractContextManager[None]:
        return nullcontext()

    def shutdown(self) -> None:
        return None


class LangfuseTracing:
    def __init__(
        self,
        *,
        client: Any,
        handler: Any,
        propagate_attributes: Any,
    ):
        self._client = client
        self._handler = handler
        self._propagate_attributes = propagate_attributes

    def callbacks(self) -> list[Any]:
        # One handler is enough: it reads the active OpenTelemetry context, so
        # the same instance nests correctly under whatever span is current.
        return [self._handler]

    @contextmanager
    def conversation(
        self,
        *,
        session_id: str,
        tags: Sequence[str],
    ) -> Iterator[None]:
        stack = ExitStack()
        try:
            stack.enter_context(
                self._client.start_as_current_observation(
                    as_type="span",
                    name=ROOT_OBSERVATION_NAME,
                )
            )
            stack.enter_context(
                self._propagate_attributes(
                    session_id=session_id,
                    tags=list(tags),
                )
            )
        except Exception as error:
            # Opening the trace failed. The request continues untraced; only
            # the entry is guarded, so failures raised by the body below still
            # propagate normally.
            logger.warning("Langfuse trace not started: %s", type(error).__name__)
            stack.close()
            yield
            return
        with stack:
            yield

    def shutdown(self) -> None:
        try:
            self._client.shutdown()
        except Exception as error:
            logger.warning(
                "Langfuse shutdown failed: %s",
                type(error).__name__,
            )


def create_tracing(
    config: Settings,
    *,
    loader: Callable[[], TracingComponents] = import_langfuse,
) -> Tracing:
    if not config.langfuse_enabled:
        return NullTracing()

    public_key = config.langfuse_public_key.get_secret_value().strip()
    secret_key = config.langfuse_secret_key.get_secret_value().strip()
    base_url = config.langfuse_base_url.strip()
    if not public_key or not secret_key or not base_url:
        logger.error(
            "Langfuse is enabled but LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY "
            "or LANGFUSE_BASE_URL is missing; continuing without tracing"
        )
        return NullTracing()

    try:
        components = loader()
    except ImportError:
        logger.error(
            "Langfuse is enabled but the package is not installed; install the "
            "'tracing' extra. Continuing without tracing"
        )
        return NullTracing()

    try:
        client = components.client_factory(
            public_key=public_key,
            secret_key=secret_key,
            base_url=base_url,
        )
        handler = components.handler_factory()
    except Exception as error:
        logger.error(
            "Langfuse client could not be created (%s); continuing without "
            "tracing",
            type(error).__name__,
        )
        return NullTracing()

    logger.info("Langfuse tracing enabled", extra={"langfuse_base_url": base_url})
    return LangfuseTracing(
        client=client,
        handler=handler,
        propagate_attributes=components.propagate_attributes,
    )

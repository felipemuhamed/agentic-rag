import logging
import sys

from app.observability.logging import JsonFormatter
from app.observability.metrics import MetricsRegistry


def test_metrics_registry_exposes_only_bounded_labels() -> None:
    metrics = MetricsRegistry()
    metrics.observe_http(status_code=200, duration_seconds=0.1)
    metrics.observe_http(status_code=401, duration_seconds=0.2)
    metrics.auth_failed()
    metrics.rate_limit_rejected()
    metrics.stream_started()
    metrics.stream_finished(outcome="completed", duration_seconds=1.5)

    payload = metrics.render_prometheus()

    assert 'status_class="2xx"} 1' in payload
    assert 'status_class="4xx"} 1' in payload
    assert 'outcome="completed"} 1' in payload
    assert "agentic_rag_streams_active 0" in payload
    assert "agentic_rag_auth_failures_total 1" in payload


def test_json_logs_omit_unsafe_fields_and_exception_text() -> None:
    try:
        raise RuntimeError("provider-internal-secret")
    except RuntimeError:
        exception_info = sys.exc_info()

    record = logging.getLogger("test").makeRecord(
        name="test",
        level=logging.ERROR,
        fn=__file__,
        lno=1,
        msg="provider operation failed",
        args=(),
        exc_info=exception_info,
        extra={"api_key": "header-private-secret", "status_code": 500},
    )

    payload = JsonFormatter().format(record)

    assert "header-private-secret" not in payload
    assert "provider-internal-secret" not in payload
    assert '"exception_type":"RuntimeError"' in payload
    assert '"status_code":500' in payload

from threading import Lock


class MetricsRegistry:
    def __init__(self):
        self._lock = Lock()
        self._http_requests: dict[str, int] = {}
        self._http_duration_count = 0
        self._http_duration_sum = 0.0
        self._stream_outcomes: dict[str, int] = {}
        self._stream_duration_count = 0
        self._stream_duration_sum = 0.0
        self._active_streams = 0
        self._auth_failures = 0
        self._rate_limit_rejections = 0

    def observe_http(self, *, status_code: int, duration_seconds: float) -> None:
        status_class = f"{status_code // 100}xx"
        with self._lock:
            self._http_requests[status_class] = (
                self._http_requests.get(status_class, 0) + 1
            )
            self._http_duration_count += 1
            self._http_duration_sum += max(duration_seconds, 0.0)

    def stream_started(self) -> None:
        with self._lock:
            self._active_streams += 1

    def stream_finished(self, *, outcome: str, duration_seconds: float) -> None:
        if outcome not in {"completed", "error", "timeout", "cancelled"}:
            raise ValueError("invalid stream outcome")
        with self._lock:
            self._active_streams = max(self._active_streams - 1, 0)
            self._stream_outcomes[outcome] = (
                self._stream_outcomes.get(outcome, 0) + 1
            )
            self._stream_duration_count += 1
            self._stream_duration_sum += max(duration_seconds, 0.0)

    def auth_failed(self) -> None:
        with self._lock:
            self._auth_failures += 1

    def rate_limit_rejected(self) -> None:
        with self._lock:
            self._rate_limit_rejections += 1

    def render_prometheus(self) -> str:
        with self._lock:
            http_requests = dict(self._http_requests)
            stream_outcomes = dict(self._stream_outcomes)
            values = {
                "http_duration_count": self._http_duration_count,
                "http_duration_sum": self._http_duration_sum,
                "stream_duration_count": self._stream_duration_count,
                "stream_duration_sum": self._stream_duration_sum,
                "active_streams": self._active_streams,
                "auth_failures": self._auth_failures,
                "rate_limit_rejections": self._rate_limit_rejections,
            }

        lines = [
            "# HELP agentic_rag_http_requests_total HTTP requests by status class.",
            "# TYPE agentic_rag_http_requests_total counter",
        ]
        for status_class in sorted(http_requests):
            lines.append(
                "agentic_rag_http_requests_total"
                f'{{status_class="{status_class}"}} {http_requests[status_class]}'
            )
        lines.extend(
            [
                "# TYPE agentic_rag_http_request_duration_seconds summary",
                "agentic_rag_http_request_duration_seconds_count "
                f"{values['http_duration_count']}",
                "agentic_rag_http_request_duration_seconds_sum "
                f"{values['http_duration_sum']:.6f}",
                "# HELP agentic_rag_streams_total Chat streams by terminal outcome.",
                "# TYPE agentic_rag_streams_total counter",
            ]
        )
        for outcome in sorted(stream_outcomes):
            lines.append(
                f'agentic_rag_streams_total{{outcome="{outcome}"}} '
                f"{stream_outcomes[outcome]}"
            )
        lines.extend(
            [
                "# TYPE agentic_rag_stream_duration_seconds summary",
                "agentic_rag_stream_duration_seconds_count "
                f"{values['stream_duration_count']}",
                "agentic_rag_stream_duration_seconds_sum "
                f"{values['stream_duration_sum']:.6f}",
                "# TYPE agentic_rag_streams_active gauge",
                f"agentic_rag_streams_active {values['active_streams']}",
                "# TYPE agentic_rag_auth_failures_total counter",
                f"agentic_rag_auth_failures_total {values['auth_failures']}",
                "# TYPE agentic_rag_rate_limit_rejections_total counter",
                "agentic_rag_rate_limit_rejections_total "
                f"{values['rate_limit_rejections']}",
            ]
        )
        return "\n".join(lines) + "\n"

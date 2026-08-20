from app.streaming.events import StreamEvent


def test_event_is_encoded_as_sse_with_utf8_json() -> None:
    event = StreamEvent(
        "status",
        {"stage": "rag_search", "message": "Buscando informação..."},
    )

    assert event.to_sse() == (
        "event: status\n"
        'data: {"stage":"rag_search","message":"Buscando informação..."}\n'
        "\n"
    )

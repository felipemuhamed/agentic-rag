from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.persistence.checkpointer import CheckpointerHandle
from app.settings import Settings


class MessageState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def compile_graph(checkpointer):
    builder = StateGraph(MessageState)
    builder.add_node(
        "answer",
        lambda state: {"messages": [AIMessage(content="ack")]},
    )
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    return builder.compile(checkpointer=checkpointer)


def test_sqlite_checkpointer_survives_runtime_restart(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "state" / "checkpoints.sqlite"
    config = Settings(
        checkpointer_mode="sqlite",
        checkpointer_path=str(checkpoint_path),
        _env_file=None,
    )
    thread = {"configurable": {"thread_id": "thread-1"}}

    first_handle = CheckpointerHandle(config)
    first_checkpointer = first_handle.start()
    assert first_checkpointer.serde._allowed_msgpack_modules is None
    first_graph = compile_graph(first_checkpointer)
    first_graph.invoke(
        {"messages": [HumanMessage(content="first")]},
        config=thread,
    )
    first_handle.close()

    second_handle = CheckpointerHandle(config)
    second_graph = compile_graph(second_handle.start())
    result = second_graph.invoke(
        {"messages": [HumanMessage(content="second")]},
        config=thread,
    )
    second_handle.close()

    assert checkpoint_path.is_file()
    assert [message.content for message in result["messages"]] == [
        "first",
        "ack",
        "second",
        "ack",
    ]

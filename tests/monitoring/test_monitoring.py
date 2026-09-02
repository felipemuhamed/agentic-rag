import asyncio

import pytest
from langchain_core.messages import AIMessage

from app.monitoring.agent import (
    MonitoringDecision,
    MonitoringInvestigator,
    MonitoringRouter,
)
from app.monitoring.bridge import AsyncBridge, AsyncBridgeClosed
from app.monitoring.client import (
    MonitoringToolset,
    MonitoringUnavailable,
    as_text,
)
from app.monitoring.models import (
    MonitoringObservation,
    is_empty_payload,
    monitoring_context,
    monitoring_documents,
    monitoring_status,
)


class FakeTool:
    def __init__(self, name: str, result: object = "ok", fails: bool = False):
        self.name = name
        self.calls: list[dict[str, object]] = []
        self._result = result
        self._fails = fails

    async def ainvoke(self, arguments: dict[str, object]) -> object:
        self.calls.append(arguments)
        if self._fails:
            raise RuntimeError("zabbix indisponível")
        return self._result


class FakeBridge:
    def run(self, coroutine, *, timeout: float):
        return asyncio.run(coroutine)


def make_toolset(tools: list[FakeTool], allowed: list[str]) -> MonitoringToolset:
    toolset = MonitoringToolset(
        connections={"monitoring": {"transport": "streamable_http", "url": "x"}},
        bridge=FakeBridge(),
        allowed_tools=allowed,
        timeout_seconds=5.0,
    )
    # Skip discovery: loading is exercised separately from dispatch.
    toolset._tools = [tool for tool in tools if tool.name in set(allowed)]
    return toolset


# --- bridge ------------------------------------------------------------------


def test_bridge_runs_a_coroutine_from_synchronous_code() -> None:
    bridge = AsyncBridge()

    async def work() -> str:
        await asyncio.sleep(0)
        return "pronto"

    try:
        assert bridge.run(work(), timeout=5) == "pronto"
    finally:
        bridge.close()


def test_bridge_times_out_instead_of_hanging_the_request() -> None:
    bridge = AsyncBridge()

    async def forever() -> None:
        await asyncio.sleep(30)

    try:
        with pytest.raises(TimeoutError):
            bridge.run(forever(), timeout=0.05)
    finally:
        bridge.close()


def test_closed_bridge_refuses_new_work() -> None:
    bridge = AsyncBridge()
    bridge.close()

    async def work() -> str:
        return "x"

    with pytest.raises(AsyncBridgeClosed):
        bridge.run(work(), timeout=1)


# --- toolset -----------------------------------------------------------------


def test_only_allowlisted_tools_are_ever_offered() -> None:
    toolset = make_toolset(
        [FakeTool("problem_get"), FakeTool("host_update")],
        allowed=["problem_get"],
    )

    # host_update exists on the server but must never reach the model.
    assert [tool.name for tool in toolset.tools()] == ["problem_get"]


def test_calling_a_tool_outside_the_allowlist_is_refused() -> None:
    toolset = make_toolset([FakeTool("problem_get")], allowed=["problem_get"])

    with pytest.raises(MonitoringUnavailable):
        toolset.call("host_update", {})


def test_tool_result_content_blocks_are_flattened() -> None:
    toolset = make_toolset(
        [
            FakeTool(
                "problem_get",
                result=[
                    {"type": "text", "text": '{"eventid": "1001"}'},
                    {"type": "text", "text": '{"eventid": "1002"}'},
                ],
            )
        ],
        allowed=["problem_get"],
    )

    content = toolset.call("problem_get", {"limit": 2})

    assert '"eventid": "1001"' in content
    assert '"eventid": "1002"' in content


def test_as_text_handles_strings_lists_and_objects() -> None:
    assert as_text("pronto") == "pronto"
    assert as_text({"type": "text", "text": "a"}) == "a"
    assert as_text(["a", {"text": "b"}]) == "a\nb"


# --- evidence ----------------------------------------------------------------


def test_observations_become_citable_documents() -> None:
    observations = [
        MonitoringObservation(
            tool="problem_get",
            arguments={"limit": 2},
            content='[{"eventid": "1001"}]',
        )
    ]

    documents = monitoring_documents(observations)

    assert documents[0].metadata["source"] == "zabbix:problem_get"
    assert documents[0].metadata["title"] == "Monitoração: problem_get"
    # The arguments travel with the evidence so a claim can be reproduced.
    assert '"limit": 2' in documents[0].content
    assert "problem_get" in monitoring_context(observations)


def test_an_empty_payload_is_recognised_whatever_shape_it_takes() -> None:
    assert is_empty_payload("[]")
    assert is_empty_payload("  {}  ")
    assert is_empty_payload("")
    assert is_empty_payload("null")
    assert not is_empty_payload('[{"eventid": "1001"}]')
    # Free text that is not JSON is data, not emptiness.
    assert not is_empty_payload("nenhum problema encontrado")


def test_an_executed_query_that_returned_nothing_says_so_in_words() -> None:
    observation = MonitoringObservation(tool="problem_get", content="[]")

    assert observation.status == "empty"
    assert "Nenhum resultado" in observation.as_document(0).content


def test_monitoring_status_separates_no_data_from_no_answer() -> None:
    data = MonitoringObservation(tool="problem_get", content='[{"eventid": "1"}]')
    empty = MonitoringObservation(tool="problem_get", content="[]")
    failed = MonitoringObservation(tool="problem_get", content="{}", status="error")

    assert monitoring_status([]) == "empty"
    assert monitoring_status([empty]) == "empty"
    assert monitoring_status([failed]) == "unavailable"
    # One useful call is enough: the question can be answered from what came.
    assert monitoring_status([empty, data]) == "ok"
    assert monitoring_status([failed, empty]) == "empty"


def test_context_is_empty_without_observations() -> None:
    assert monitoring_context([]) == ""
    assert monitoring_documents([]) == []


# --- router ------------------------------------------------------------------


class FakeStructuredModel:
    def __init__(self, decision: MonitoringDecision):
        self.decision = decision
        self.calls: list[object] = []

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        return self.decision


def test_router_reports_when_live_data_is_needed() -> None:
    model = FakeStructuredModel(
        MonitoringDecision(needs_monitoring=True, reason="pergunta sobre alertas")
    )

    decision = MonitoringRouter(model).decide("quais os últimos alertas?")

    assert decision.needs_monitoring is True


def test_router_rejects_a_blank_question() -> None:
    model = FakeStructuredModel(
        MonitoringDecision(needs_monitoring=False, reason="x")
    )

    with pytest.raises(ValueError):
        MonitoringRouter(model).decide("   ")


# --- investigator ------------------------------------------------------------


class ScriptedModel:
    """Emit a scripted sequence of assistant turns, with or without tool calls."""

    def __init__(self, turns: list[AIMessage]):
        self.turns = turns
        self.bound_tools: list[object] = []
        self.invocations = 0

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return self

    def invoke(self, messages):
        turn = self.turns[min(self.invocations, len(self.turns) - 1)]
        self.invocations += 1
        return turn


def tool_call(name: str, args: dict[str, object], call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def test_investigator_collects_observations_and_stops() -> None:
    tool = FakeTool("problem_get", result="[]")
    toolset = make_toolset([tool], allowed=["problem_get"])
    model = ScriptedModel(
        [
            tool_call("problem_get", {"limit": 5}, "c1"),
            AIMessage(content="coletado"),
        ]
    )

    observations = MonitoringInvestigator(
        model, toolset=toolset, max_iterations=3
    ).investigate("quais os últimos alertas?")

    assert [item.tool for item in observations] == ["problem_get"]
    assert tool.calls == [{"limit": 5}]
    assert model.invocations == 2


def test_investigator_stops_at_the_iteration_budget() -> None:
    tool = FakeTool("problem_get", result="[]")
    toolset = make_toolset([tool], allowed=["problem_get"])
    # A model that never stops asking for tools.
    model = ScriptedModel([tool_call("problem_get", {}, "c1")])

    observations = MonitoringInvestigator(
        model, toolset=toolset, max_iterations=2
    ).investigate("pergunta")

    assert len(observations) == 2
    assert model.invocations == 2


def test_a_failing_tool_is_recorded_rather_than_raised() -> None:
    toolset = make_toolset(
        [FakeTool("problem_get", fails=True)], allowed=["problem_get"]
    )
    model = ScriptedModel(
        [tool_call("problem_get", {}, "c1"), AIMessage(content="fim")]
    )

    observations = MonitoringInvestigator(
        model, toolset=toolset, max_iterations=2
    ).investigate("pergunta")

    # The generator must see that the datum is missing instead of getting a
    # confident answer built on nothing.
    assert "erro" in observations[0].content
    # And it is an error, not an empty result: the two lead to different answers.
    assert observations[0].status == "error"


def test_investigator_without_tools_reports_unavailable() -> None:
    toolset = make_toolset([], allowed=[])
    model = ScriptedModel([AIMessage(content="x")])

    with pytest.raises(MonitoringUnavailable):
        MonitoringInvestigator(model, toolset=toolset).investigate("pergunta")

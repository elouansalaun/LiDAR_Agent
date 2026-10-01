"""The agent as a LangGraph state machine.

    START -> agent --(tool calls?)--> tools -> validator --> agent -> ... -> END
                   \\--(final answer)--> END

agent      asks Claude what to do next (call tools, or answer)
tools      runs the requested tools (plain Python functions from src/tools)
validator  checks the tool outputs; sends them back to Claude, adding
           "VALIDATOR" feedback when something is implausible (see validator.py)

The conversation is stored in the Anthropic Messages format, so the agent
node is a single API call with no conversion layer.
"""

from __future__ import annotations

import json
import operator
import time
from dataclasses import dataclass, field
from typing import Annotated, Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from src import config
from src.agent.prompts import SYSTEM_PROMPT
from src.agent.validator import validate_results
from src.tools import TOOLS, tool_schemas

# Tool outputs worth remembering for the UI (latest value wins).
ARTIFACT_KEYS = ("point_cloud_path", "buildings_path", "map_path", "area_bbox_lonlat")


def merge_dicts(old: dict, new: dict) -> dict:
    return {**old, **new}


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]       # full conversation (Anthropic format)
    pending_results: list                         # tool outputs waiting for validation
    steps: int                                    # number of LLM calls in this turn
    validation_retries: int
    stop_reason: str
    artifacts: Annotated[dict, merge_dicts]       # latest file paths produced by tools
    trace: Annotated[list, operator.add]          # what happened, for the CLI / UI


def build_graph(llm: Callable, max_steps: int = 25):
    """Wire the three nodes together. `llm(system, messages, tools)` returns a Claude message."""

    def agent_node(state: AgentState) -> dict:
        response = llm(SYSTEM_PROMPT, state["messages"], tool_schemas())
        # Text written alongside tool calls is progress narration; the final answer is returned separately.
        texts = [b.text for b in response.content if b.type == "text" and b.text.strip()] \
            if response.stop_reason == "tool_use" else []
        return {
            # Append the content blocks unchanged (thinking blocks included): the API needs them back as-is.
            "messages": [{"role": "assistant", "content": response.content}],
            "steps": state["steps"] + 1,
            "stop_reason": response.stop_reason,
            "trace": [{"type": "assistant_text", "text": t} for t in texts],
        }

    def tools_node(state: AgentState) -> dict:
        tool_calls = [b for b in state["messages"][-1]["content"] if b.type == "tool_use"]
        results, trace, artifacts = [], [], {}
        for call in tool_calls:
            start = time.perf_counter()
            if call.name in TOOLS:
                output, is_error = TOOLS[call.name].run_json(call.input)
            else:
                output, is_error = json.dumps({"error": f"Unknown tool '{call.name}'"}), True

            results.append({"tool_use_id": call.id, "name": call.name, "output": output, "is_error": is_error})
            trace.append({"type": "tool_call", "name": call.name, "input": call.input, "output": json.loads(output),
                          "is_error": is_error, "seconds": round(time.perf_counter() - start, 1)})
            if not is_error:
                artifacts.update({k: v for k, v in json.loads(output).items() if k in ARTIFACT_KEYS})
        return {"pending_results": results, "trace": trace, "artifacts": artifacts}

    def validator_node(state: AgentState) -> dict:
        results = state["pending_results"]
        content: list[dict[str, Any]] = [
            {"type": "tool_result", "tool_use_id": r["tool_use_id"], "content": r["output"], "is_error": r["is_error"]}
            for r in results
        ]
        report = validate_results([(r["name"], r["output"], r["is_error"]) for r in results])

        retries = state["validation_retries"]
        trace = []
        if report.issues:
            can_retry = retries < config.MAX_VALIDATION_RETRIES
            if not report.is_valid and can_retry:
                retries += 1
            feedback = report.feedback(can_retry=can_retry)
            # The feedback travels in the same user message as the tool results.
            content.append({"type": "text", "text": feedback})
            trace.append({"type": "validation", "valid": report.is_valid, "feedback": feedback})

        return {
            "messages": [{"role": "user", "content": content}],
            "pending_results": [],
            "validation_retries": retries,
            "trace": trace,
        }

    def after_agent(state: AgentState) -> str:
        return "tools" if state["stop_reason"] == "tool_use" else END

    def after_validator(state: AgentState) -> str:
        return "agent" if state["steps"] < max_steps else END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("validator", validator_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", after_agent, ["tools", END])
    graph.add_edge("tools", "validator")
    graph.add_conditional_edges("validator", after_validator, ["agent", END])
    return graph.compile()


@dataclass
class AgentResult:
    answer: str
    trace: list[dict] = field(default_factory=list)
    artifacts: dict = field(default_factory=dict)
    steps: int = 0


class LidarAgent:
    """Multi-turn wrapper: remembers the conversation between questions."""

    def __init__(self, llm: Callable | None = None, max_steps: int = 25):
        if llm is None:
            from src.agent.llm import (
                make_llm,  # imported lazily so tests do not need the SDK configured
            )

            llm = make_llm()
        self.llm = llm
        self.max_steps = max_steps
        self.graph = build_graph(llm, max_steps)
        self.messages: list = []
        self.artifacts: dict = {}

    def ask(self, question: str, on_event: Callable[[dict], None] | None = None) -> AgentResult:
        state: AgentState = {
            "messages": [*self.messages, {"role": "user", "content": question}],
            "pending_results": [],
            "steps": 0,
            "validation_retries": 0,
            "stop_reason": "",
            "artifacts": {},
            "trace": [],
        }
        final_state = state
        run_config = {"recursion_limit": 3 * self.max_steps + 5}
        for mode, chunk in self.graph.stream(state, config=run_config, stream_mode=["updates", "values"]):
            if mode == "values":
                final_state = chunk
            elif on_event:
                for update in chunk.values():
                    for event in (update or {}).get("trace", []):
                        on_event(event)

        self.messages = final_state["messages"]
        self.artifacts.update(final_state["artifacts"])
        return AgentResult(
            answer=_final_answer(final_state, self.max_steps),
            trace=final_state["trace"],
            artifacts=dict(self.artifacts),
            steps=final_state["steps"],
        )


def _final_answer(state: AgentState, max_steps: int) -> str:
    last = state["messages"][-1]
    if last["role"] != "assistant":
        return f"Stopped after {max_steps} steps without a final answer. See the trace for the partial results."
    text = "\n".join(b.text for b in last["content"] if b.type == "text").strip()
    if state["stop_reason"] == "refusal":
        return text or "The model declined to answer this request."
    if state["stop_reason"] == "max_tokens":
        return text + "\n\n[answer truncated: max_tokens reached]"
    return text

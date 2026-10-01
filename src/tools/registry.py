"""A tiny tool framework: turn a typed Python function into an LLM tool.

    @tool
    def normalize_height(point_cloud_path: Annotated[str, Field(description="...")]) -> dict:
        '''Short description the LLM will read.'''

The decorator:
  * builds a JSON schema from the type hints (via a pydantic model),
  * validates the arguments sent by the LLM before calling the function,
  * turns any exception into a readable error message the agent can react to,
  * registers the tool in TOOLS so the agent can find it by name.

We could have used LangChain's @tool, but ~80 lines of our own code keep the
project dependency-light and make it obvious what the LLM actually sees.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from typing import Any, Callable, get_type_hints

from pydantic import BaseModel, ValidationError, create_model


class ToolError(Exception):
    """An expected failure with a message meant for the LLM (e.g. "run X first")."""


@dataclass
class Tool:
    name: str
    description: str
    function: Callable[..., dict]
    arguments_model: type[BaseModel]

    def schema(self) -> dict:
        """Tool definition in the format expected by the Claude Messages API."""
        input_schema = self.arguments_model.model_json_schema()
        input_schema.pop("title", None)
        for prop in input_schema.get("properties", {}).values():
            prop.pop("title", None)
        return {"name": self.name, "description": self.description, "input_schema": input_schema}

    def run(self, arguments: dict[str, Any]) -> dict:
        """Validate the arguments, call the function and return its result.

        Raises ToolError with an LLM-friendly message on any failure.
        """
        try:
            validated = self.arguments_model(**(arguments or {}))
        except ValidationError as error:
            problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors())
            raise ToolError(f"Invalid arguments for {self.name}: {problems}") from error

        try:
            return self.function(**validated.model_dump())
        except ToolError:
            raise
        except (FileNotFoundError, ValueError) as error:
            raise ToolError(str(error)) from error
        except Exception as error:  # unexpected bug: still report it instead of crashing the agent
            raise ToolError(f"{type(error).__name__}: {error}") from error

    def run_json(self, arguments: dict[str, Any]) -> tuple[str, bool]:
        """Same as run() but returns (json_text, is_error) for the agent loop."""
        try:
            return json.dumps(self.run(arguments), default=str), False
        except ToolError as error:
            return json.dumps({"error": str(error)}), True


TOOLS: dict[str, Tool] = {}


def tool(function: Callable[..., dict]) -> Callable[..., dict]:
    """Register `function` as an agent tool. The function stays callable as usual."""
    hints = get_type_hints(function, include_extras=True)  # keeps Annotated[...] descriptions
    fields = {}
    for name, parameter in inspect.signature(function).parameters.items():
        default = ... if parameter.default is inspect.Parameter.empty else parameter.default
        fields[name] = (hints[name], default)

    arguments_model = create_model(f"{function.__name__}_arguments", **fields)
    TOOLS[function.__name__] = Tool(
        name=function.__name__,
        description=inspect.getdoc(function) or "",
        function=function,
        arguments_model=arguments_model,
    )
    return function


def tool_schemas() -> list[dict]:
    """All registered tool definitions, sorted by name so the prompt stays byte-identical (prompt caching)."""
    return [TOOLS[name].schema() for name in sorted(TOOLS)]

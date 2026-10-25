"""Command-line agent.
Without API credentials the offline demo planner is used automatically.
"""

from __future__ import annotations

import argparse
import json
import sys

import anthropic

from src.agent.graph import LidarAgent
from src.agent.llm import has_credentials, make_llm


def print_event(event: dict) -> None:
    """Live progress: what the agent is doing, one line per step."""
    if event["type"] == "assistant_text":
        print(f"  · {event['text']}", flush=True)
    elif event["type"] == "tool_call":
        arguments = ", ".join(f"{k}={json.dumps(v)}" for k, v in event["input"].items())
        status = "error" if event["is_error"] else "ok"
        print(f"  -> {event['name']}({arguments})  [{status}, {event['seconds']}s]", flush=True)
        if event["is_error"]:
            print(f"     {event['output'].get('error')}", flush=True)
    elif event["type"] == "validation":
        label = "validator: OK with warnings" if event["valid"] else "validator: REJECTED"
        print(f"  !! {label}", flush=True)
        for line in event["feedback"].splitlines():
            print(f"     {line}", flush=True)


def ask(agent: LidarAgent, question: str) -> None:
    try:
        result = agent.ask(question, on_event=print_event)
    except anthropic.AuthenticationError:
        sys.exit("Claude API authentication failed. Put ANTHROPIC_API_KEY=... in a .env file (see .env.example).")
    except anthropic.APIConnectionError:
        sys.exit("Could not reach the Claude API. Check your internet connection.")
    except anthropic.APIStatusError as error:
        sys.exit(f"Claude API error {error.status_code}: {error.message}")
    print("\n" + result.answer + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions about the Montreal LiDAR tiles.")
    parser.add_argument("question", nargs="*", help="question to ask (omit for an interactive session)")
    parser.add_argument("--offline", action="store_true", help="use the rule-based demo planner instead of Claude")
    args = parser.parse_args()

    if not args.offline and not has_credentials():
        print("No Claude API credentials found (set ANTHROPIC_API_KEY in .env).\n"
              "Using the offline demo planner: rule-based, no LLM, building and coverage questions only.\n")
    agent = LidarAgent(llm=make_llm(offline=args.offline))
    if args.question:
        ask(agent, " ".join(args.question))
        return

    print("LiDAR agent - ask a question about the Montreal tiles (empty line or Ctrl-D to quit).")
    print('Example: "Find buildings taller than 20 meters around Université de Montréal."\n')
    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        ask(agent, question)


if __name__ == "__main__":
    main()

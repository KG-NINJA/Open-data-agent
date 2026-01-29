"""CLI entrypoint for the data agent."""
from __future__ import annotations

import argparse
import os
import sys

from .agent import DataAgent
from .data_loader import load_catalog
from .log_store import LogStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Natural language data agent")
    parser.add_argument(
        "--data",
        nargs="*",
        help="Paths to CSV/JSON/SQLite files or directories",
    )
    parser.add_argument(
        "--question",
        help="Single question to answer (omit for interactive mode)",
    )
    parser.add_argument(
        "--log",
        default=os.path.join("data_agent", "logs", "agent_logs.jsonl"),
        help="Path to JSONL log file",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="OpenAI model name (falls back to OPENAI_MODEL)",
    )
    parser.add_argument(
        "--show-plan",
        action="store_true",
        help="Print the generated query plan",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_paths = args.data or []
    if not data_paths:
        user_input = input("Data paths (comma-separated): ").strip()
        if user_input:
            data_paths = [item.strip() for item in user_input.split(",") if item.strip()]
    if not data_paths:
        print("No data sources provided.")
        sys.exit(1)

    catalog = load_catalog(data_paths)
    if not catalog.list_tables():
        print("No supported data sources found.")
        sys.exit(1)

    log_store = LogStore(args.log)
    agent = DataAgent(catalog, log_store=log_store, model=args.model)

    if args.question:
        response = agent.answer(args.question)
        _print_response(response, show_plan=args.show_plan)
        return

    print("Data agent ready. Type 'exit' to quit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        response = agent.answer(question)
        _print_response(response, show_plan=args.show_plan)


def _print_response(response, show_plan: bool = False) -> None:
    print("\nSummary:")
    print(response.summary)
    if response.warnings:
        print("\nWarnings:")
        for warning in response.warnings:
            print(f"- {warning}")
    print("\nPreview:")
    print(response.preview)
    if show_plan:
        print("\nPlan:")
        print(response.plan)


if __name__ == "__main__":
    main()

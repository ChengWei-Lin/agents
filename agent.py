from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request


OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://localhost:11434/api/chat")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
MEMORY_PATH = Path(__file__).with_name("memory.json")


SYSTEM_PROMPT = """
You are a practical personal coding and productivity agent.

Use tools when they help you answer accurately or preserve important user context.
When a user shares a durable preference, project fact, or decision they are likely to want later,
consider saving it with save_memory.

Keep answers concise and useful.
""".strip()


class MemoryStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def _load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []

    def _save(self, records: list[dict[str, Any]]) -> None:
        self.path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    def add(self, category: str, content: str) -> dict[str, Any]:
        records = self._load()
        item = {
            "id": len(records) + 1,
            "category": category,
            "content": content,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        records.append(item)
        self._save(records)
        return item

    def list(self) -> list[dict[str, Any]]:
        return self._load()

    def search(self, query: str) -> list[dict[str, Any]]:
        needle = query.lower()
        return [
            item
            for item in self._load()
            if needle in item["category"].lower() or needle in item["content"].lower()
        ]


@dataclass
class ToolResult:
    name: str
    output: str


class AgentTools:
    def __init__(self, memory: MemoryStore) -> None:
        self.memory = memory

    def schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Get the current local and UTC time.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "save_memory",
                    "description": "Save a durable user preference, fact, or project decision for later reuse.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "description": "Short label such as preference, project, or decision.",
                            },
                            "content": {
                                "type": "string",
                                "description": "The memory text to store.",
                            },
                        },
                        "required": ["category", "content"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_memories",
                    "description": "List all stored memories.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_memories",
                    "description": "Search stored memories by keyword.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Keyword or phrase to search in stored memories.",
                            }
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            },
        ]

    def call(self, name: str, args: dict[str, Any]) -> ToolResult:
        if name == "get_time":
            local_now = datetime.now().astimezone().isoformat()
            utc_now = datetime.now(timezone.utc).isoformat()
            return ToolResult(name, json.dumps({"local_time": local_now, "utc_time": utc_now}))
        if name == "save_memory":
            item = self.memory.add(category=args["category"], content=args["content"])
            return ToolResult(name, json.dumps(item))
        if name == "list_memories":
            return ToolResult(name, json.dumps(self.memory.list()))
        if name == "search_memories":
            return ToolResult(name, json.dumps(self.memory.search(query=args["query"])))
        raise ValueError(f"Unknown tool: {name}")


def create_response(payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        OLLAMA_API_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def assistant_text(message: dict[str, Any]) -> str:
    return str(message.get("content", "")).strip()


def run_agent_turn(
    user_text: str,
    tools: AgentTools,
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    messages.append({"role": "user", "content": user_text})

    while True:
        response = create_response(
            {
                "model": DEFAULT_MODEL,
                "messages": messages,
                "tools": tools.schemas(),
                "stream": False,
                "think": False,
            }
        )
        message = response["message"]
        messages.append(message)

        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            return assistant_text(message), messages

        for tool_call in tool_calls:
            function = tool_call.get("function", {})
            name = function.get("name")
            args = function.get("arguments", {})
            if not isinstance(args, dict):
                args = {}
            result = tools.call(str(name), args)
            messages.append(
                {
                    "role": "tool",
                    "tool_name": result.name,
                    "content": result.output,
                }
            )


def main() -> int:
    memory = MemoryStore(MEMORY_PATH)
    tools = AgentTools(memory)
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    print(f"Starter Agent running with model: {DEFAULT_MODEL}")
    print(f"Ollama endpoint: {OLLAMA_API_URL}")
    print("Type 'exit' or 'quit' to stop.")

    while True:
        try:
            user_text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return 0

        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            print("Bye.")
            return 0

        try:
            answer, messages = run_agent_turn(
                user_text=user_text,
                tools=tools,
                messages=messages,
            )
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            print(f"\nAPI error: {exc.code} {detail}")
            continue
        except error.URLError as exc:
            print(f"\nNetwork error: {exc}")
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"\nUnexpected error: {exc}")
            continue

        print(f"\nAgent: {answer}")


if __name__ == "__main__":
    raise SystemExit(main())

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
MEMORIES_DIR = Path(__file__).with_name("memories")
ENV_FILE_PATH = Path(__file__).with_name(".env.local")
PROMPTS_DIR = Path(__file__).with_name("prompts")
PROFILE_SEED_PATH = Path(__file__).with_name("user_profile_seed.json")
DEFAULT_PERSONA = "mnemosyne"


@dataclass(frozen=True)
class Persona:
    key: str
    label: str
    instructions: str
    instructions_path: Path | None = None


@dataclass(frozen=True)
class TutorMode:
    key: str
    label: str
    instructions: str


PERSONAS: dict[str, Persona] = {
    "mnemosyne": Persona(
        key="mnemosyne",
        label="Mnemosyne",
        instructions="You are Mnemosyne, a warm general knowledge keeper for the user.",
        instructions_path=PROMPTS_DIR / "mnemosyne.md",
    ),
    "language_tutor": Persona(
        key="language_tutor",
        label="Language Tutor",
        instructions="You are a patient Chinese and German learning coach.",
        instructions_path=PROMPTS_DIR / "language_tutor.md",
    ),
    "general": Persona(
        key="general",
        label="General Assistant",
        instructions="""
You are a practical personal assistant for everyday questions and lightweight tasks.
Be concise, helpful, and flexible across a wide range of topics.
""".strip(),
    ),
}
DEFAULT_TUTOR_MODE = "conversation"

TUTOR_MODES: dict[str, TutorMode] = {
    "conversation": TutorMode(
        key="conversation",
        label="Conversation",
        instructions="""
Default to natural back-and-forth practice in the target language.
Keep replies slightly above the user's current level when possible, then explain difficult parts only when useful.
""".strip(),
    ),
    "correction": TutorMode(
        key="correction",
        label="Correction",
        instructions="""
Treat the user's input as learner output to correct.
Return a corrected version first, then briefly explain the main mistakes and offer one improved example sentence.
""".strip(),
    ),
    "translation": TutorMode(
        key="translation",
        label="Translation",
        instructions="""
Prioritize accurate translation with brief notes about nuance, register, or alternative phrasings when they matter.
If the source language is ambiguous, say so instead of guessing silently.
""".strip(),
    ),
    "drill": TutorMode(
        key="drill",
        label="Drill",
        instructions="""
Lead short guided practice.
Give one focused exercise at a time, wait for the user's answer, then give feedback and the next step.
""".strip(),
    ),
    "quiz": TutorMode(
        key="quiz",
        label="Quiz",
        instructions="""
Act as a light quizmaster for vocabulary, grammar, or translation checks.
Ask concise questions, avoid giving away the answer too early, and score or summarize performance when appropriate.
""".strip(),
    ),
}


BASE_SYSTEM_PROMPT = """
Use tools when they help you answer accurately or preserve important user context.
When a user shares a durable preference, project fact, life detail, or decision they are likely
to want later, consider saving it with save_memory.

Keep answers concise and useful.
""".strip()


def load_env_file(path: Path = ENV_FILE_PATH) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


load_env_file()


def load_text_if_exists(path: Path | None, fallback: str) -> str:
    if path is None or not path.exists():
        return fallback
    return path.read_text(encoding="utf-8").strip()


def load_profile_seed(path: Path = PROFILE_SEED_PATH) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if isinstance(loaded, dict):
        return loaded
    return {}


def format_profile_seed(profile: dict[str, Any]) -> str:
    if not profile:
        return ""
    return json.dumps(profile, indent=2, ensure_ascii=False)


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

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()


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


class LocalAgentSession:
    def __init__(
        self,
        memory_path: Path,
        persona: str = DEFAULT_PERSONA,
        tutor_mode: str = DEFAULT_TUTOR_MODE,
    ) -> None:
        self.memory = MemoryStore(memory_path)
        self.tools = AgentTools(self.memory)
        self.persona_key = normalize_persona(persona)
        self.tutor_mode = normalize_tutor_mode(tutor_mode)
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": build_system_prompt(self.persona_key, self.tutor_mode)}
        ]

    def run(self, user_text: str) -> str:
        answer, self.messages = run_agent_turn(
            user_text=user_text,
            tools=self.tools,
            messages=self.messages,
        )
        return answer

    def reset_history(self) -> None:
        self.messages = [{"role": "system", "content": build_system_prompt(self.persona_key, self.tutor_mode)}]

    def switch_persona(self, persona: str) -> None:
        self.persona_key = normalize_persona(persona)
        self.reset_history()

    def switch_tutor_mode(self, tutor_mode: str) -> None:
        self.tutor_mode = normalize_tutor_mode(tutor_mode)
        self.reset_history()

    def forget_all(self) -> None:
        self.reset_history()
        self.memory.clear()

    def memory_count(self) -> int:
        return len(self.memory.list())

    def persona_label(self) -> str:
        return PERSONAS[self.persona_key].label

    def tutor_mode_label(self) -> str:
        return TUTOR_MODES[self.tutor_mode].label


def normalize_persona(persona: str | None) -> str:
    if not persona:
        return DEFAULT_PERSONA
    key = persona.strip().lower().replace("-", "_").replace(" ", "_")
    return key if key in PERSONAS else DEFAULT_PERSONA


def normalize_tutor_mode(tutor_mode: str | None) -> str:
    if not tutor_mode:
        return DEFAULT_TUTOR_MODE
    key = tutor_mode.strip().lower().replace("-", "_").replace(" ", "_")
    return key if key in TUTOR_MODES else DEFAULT_TUTOR_MODE


def build_system_prompt(persona: str, tutor_mode: str = DEFAULT_TUTOR_MODE) -> str:
    chosen = PERSONAS[normalize_persona(persona)]
    instructions = load_text_if_exists(chosen.instructions_path, chosen.instructions)
    prompt_parts = [instructions, BASE_SYSTEM_PROMPT]

    if chosen.key == "language_tutor":
        mode = TUTOR_MODES[normalize_tutor_mode(tutor_mode)]
        prompt_parts.append(f"Current tutor mode: {mode.label} (`{mode.key}`)\n{mode.instructions}")

    profile_seed = load_profile_seed()
    if profile_seed:
        prompt_parts.append(
            "Structured user profile seed:\n"
            "Treat this as editable working context. Preserve nuance, avoid treating every line as immutable fact, "
            "and update your understanding when the user corrects or deepens it.\n"
            f"{format_profile_seed(profile_seed)}"
        )

    return "\n\n".join(prompt_parts)


def memory_path_for_identity(identity: str) -> Path:
    MEMORIES_DIR.mkdir(parents=True, exist_ok=True)
    safe_identity = "".join(ch for ch in identity if ch.isalnum() or ch in {"-", "_"})
    if not safe_identity:
        safe_identity = "default"
    return MEMORIES_DIR / f"{safe_identity}.json"


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
    session = LocalAgentSession(MEMORY_PATH)

    print(f"Starter Agent running with model: {DEFAULT_MODEL}")
    print(f"Ollama endpoint: {OLLAMA_API_URL}")
    print("Type 'exit' or 'quit' to stop.")
    print(f"Available personas: {', '.join(sorted(PERSONAS))}")
    print("Use '/persona <name>' to switch personas and reset chat history for that persona.")
    print(f"Language tutor modes: {', '.join(sorted(TUTOR_MODES))}")
    print("Use '/mode <name>' while in language_tutor to switch lesson mode and reset tutor chat history.")

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
        if user_text.lower().startswith("/persona "):
            requested = user_text.split(" ", 1)[1].strip()
            session.switch_persona(requested)
            print(f"\nAgent persona switched to: {session.persona_label()} ({session.persona_key})")
            continue
        if user_text.lower().startswith("/mode "):
            requested = user_text.split(" ", 1)[1].strip()
            session.switch_tutor_mode(requested)
            print(f"\nTutor mode switched to: {session.tutor_mode_label()} ({session.tutor_mode})")
            continue

        try:
            answer = session.run(user_text)
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

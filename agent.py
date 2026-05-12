from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib import error, request

from pypinyin import Style, lazy_pinyin


MEMORY_PATH = Path(__file__).with_name("memory.json")
MEMORIES_DIR = Path(__file__).with_name("memories")
ENV_FILE_PATH = Path(__file__).with_name(".env.local")
PROMPTS_DIR = Path(__file__).with_name("prompts")
PROFILE_SEED_PATH = Path(__file__).with_name("user_profile_seed.json")
DEFAULT_PERSONA = "mnemosyne"
CHINESE_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
CHINESE_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")
PINYIN_PAREN_RE = re.compile(r"\([^()]*[āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜüńňǹĀÁǍÀĒÉĚÈĪÍǏÌŌÓǑÒŪÚǓÙǕǗǙǛÜ][^()]*\)")
CHINESE_LESSON_INTENT_RE = re.compile(
    r"\b("
    r"teach|learn|lesson|vocab|vocabulary|phrase|phrases|greeting|greetings|"
    r"beginner|chinese|mandarin|taiwanese mandarin|say|translate"
    r")\b",
    re.IGNORECASE,
)
SIMPLIFIED_TO_TRADITIONAL = str.maketrans(
    {
        "汉": "漢",
        "语": "語",
        "学": "學",
        "习": "習",
        "请": "請",
        "谢": "謝",
        "对": "對",
        "吗": "嗎",
        "这": "這",
        "个": "個",
        "什": "什",
        "么": "麼",
        "见": "見",
    }
)


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


@dataclass(frozen=True)
class ChineseTeachingEntry:
    phrase: str
    glosses: list[str]
    meaning: str


COMMON_CHINESE_LESSON_ENTRIES: dict[str, ChineseTeachingEntry] = {
    "你好": ChineseTeachingEntry(
        phrase="你好",
        glosses=["你 You", "好 Good"],
        meaning='"Hello." Used as a basic greeting.',
    ),
    "謝謝": ChineseTeachingEntry(
        phrase="謝謝",
        glosses=["謝 Thank"],
        meaning='"Thank you." Used to express thanks.',
    ),
    "請": ChineseTeachingEntry(
        phrase="請",
        glosses=["請 Please"],
        meaning='"Please." Used to make a request polite.',
    ),
    "再見": ChineseTeachingEntry(
        phrase="再見",
        glosses=["再 Again", "見 See"],
        meaning='"Goodbye." Literally "see again."',
    ),
    "早安": ChineseTeachingEntry(
        phrase="早安",
        glosses=["早 Early", "安 Peace"],
        meaning='"Good morning." Used as a morning greeting.',
    ),
}

ENGLISH_TO_CHINESE_PHRASES: dict[str, list[str]] = {
    "hello": ["你好"],
    "hi": ["你好"],
    "greeting": ["你好"],
    "greetings": ["你好", "早安"],
    "thank": ["謝謝"],
    "thanks": ["謝謝"],
    "thank you": ["謝謝"],
    "please": ["請"],
    "goodbye": ["再見"],
    "bye": ["再見"],
    "good morning": ["早安"],
}


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
OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://localhost:11434/api/chat")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")


def configure_stdio() -> None:
    for stream_name in ("stdin", "stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


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
        if self.persona_key == "language_tutor" and is_chinese_lesson_request(user_text):
            answer = render_chinese_lesson_request(user_text)
            self.messages.append({"role": "user", "content": user_text})
            self.messages.append({"role": "assistant", "content": answer})
            return answer

        answer, self.messages = run_agent_turn(
            user_text=user_text,
            tools=self.tools,
            messages=self.messages,
        )
        return postprocess_answer(answer, self.persona_key)

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


@lru_cache(maxsize=8)
def model_supports_tools(model_name: str) -> bool:
    payload = {"name": model_name}
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        OLLAMA_API_URL.replace("/api/chat", "/api/show"),
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=30) as resp:
        details = json.loads(resp.read().decode("utf-8"))
    capabilities = details.get("capabilities") or []
    return "tools" in capabilities


def assistant_text(message: dict[str, Any]) -> str:
    return str(message.get("content", "")).strip()


def text_contains_chinese(text: str) -> bool:
    return bool(CHINESE_CHAR_RE.search(text))


def normalize_traditional_chinese(text: str) -> str:
    return text.translate(SIMPLIFIED_TO_TRADITIONAL)


def is_chinese_lesson_request(text: str) -> bool:
    normalized = text.strip().lower()
    if not normalized:
        return False
    if text_contains_chinese(normalized):
        return len("".join(CHINESE_RUN_RE.findall(normalized))) <= 8 or bool(
            CHINESE_LESSON_INTENT_RE.search(normalized)
        )
    if "chinese" in normalized or "mandarin" in normalized:
        return bool(CHINESE_LESSON_INTENT_RE.search(normalized))
    if re.search(r"\b(how do i say|how to say|what is|translate)\b", normalized):
        return any(text_contains_english_phrase(normalized, key) for key in ENGLISH_TO_CHINESE_PHRASES)
    return any(
        phrase in normalized
        for phrase in (
            "teach me chinese",
            "teach me mandarin",
            "beginner chinese",
            "chinese lesson",
            "mandarin lesson",
        )
    )


def text_contains_english_phrase(text: str, phrase: str) -> bool:
    return bool(re.search(rf"\b{re.escape(phrase)}\b", text))


def requested_chinese_lesson_phrases(text: str, limit: int = 2) -> list[str]:
    phrases: list[str] = []
    normalized_text = normalize_traditional_chinese(text)
    for phrase in candidate_chinese_phrases(normalized_text, limit=limit):
        if phrase not in phrases:
            phrases.append(phrase)

    lowered = text.lower()
    for english, mapped_phrases in ENGLISH_TO_CHINESE_PHRASES.items():
        if text_contains_english_phrase(lowered, english):
            for phrase in mapped_phrases:
                if phrase not in phrases:
                    phrases.append(phrase)
                if len(phrases) >= limit:
                    return phrases

    if not phrases and ("greeting" in lowered or "hello" in lowered or "hi" in lowered):
        phrases.append("你好")
    if not phrases and ("chinese" in lowered or "mandarin" in lowered):
        phrases.append("你好")
    return phrases[:limit]


def deterministic_chinese_lesson_entry(phrase: str) -> ChineseTeachingEntry:
    traditional_phrase = normalize_traditional_chinese(phrase)
    known = COMMON_CHINESE_LESSON_ENTRIES.get(traditional_phrase)
    if known:
        return known
    glosses = [f"{char} Character" for char in traditional_phrase]
    return ChineseTeachingEntry(
        phrase=traditional_phrase,
        glosses=glosses,
        meaning="A Chinese phrase. Ask for a deeper explanation when you want nuance or examples.",
    )


def render_chinese_lesson_request(text: str) -> str:
    phrases = requested_chinese_lesson_phrases(text)
    if not phrases:
        phrases = ["你好"]
    entries = [deterministic_chinese_lesson_entry(phrase) for phrase in phrases]
    return "\n\n".join(render_chinese_teaching_entry(entry) for entry in entries)


def zhuyin_syllables(text: str) -> list[str]:
    return lazy_pinyin(
        text,
        style=Style.BOPOMOFO,
        tone_sandhi=True,
        errors=lambda chars: list(chars),
    )


def pinyin_syllables(text: str) -> list[str]:
    return lazy_pinyin(
        text,
        style=Style.TONE,
        tone_sandhi=True,
        errors=lambda chars: list(chars),
    )


def strip_parenthetical_pinyin(text: str) -> str:
    cleaned = PINYIN_PAREN_RE.sub("", text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    return cleaned.strip()


def reformat_chinese_teaching_response(text: str) -> str:
    payload = {
        "model": DEFAULT_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a strict formatter for Chinese language-learning examples.\n"
                    "Rewrite the content into compact beginner-friendly teaching examples only.\n"
                    "Use English for all meanings and explanations.\n"
                    "For each phrase, use exactly this shape:\n"
                    "PHRASE\n"
                    "Meaning:\n"
                    "- CHARACTER_OR_WORD_GLOSS\n"
                    "- WHOLE_PHRASE_MEANING_AND_USE\n"
                    "Rules:\n"
                    "- Use Traditional Chinese for the phrase.\n"
                    "- Use English glosses, for example: 你 (you) 好 (good).\n"
                    "- Keep the meaning section in English, not Chinese.\n"
                    "- Use a natural whole-phrase meaning at the end, for example: 你好 \"Hello.\" Used to greet one person.\n"
                    "- No introductions, no conclusions, and no extra commentary.\n"
                    "- Keep to at most two phrases unless the user asked for more."
                ),
            },
            {
                "role": "user",
                "content": text,
            },
        ],
        "stream": False,
        "think": False,
    }
    response = create_response(payload)
    return assistant_text(response["message"])


def extract_json_block(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def parse_reformatted_teaching_entries(text: str) -> list[ChineseTeachingEntry]:
    entries: list[ChineseTeachingEntry] = []
    current_phrase = ""
    current_glosses: list[str] = []
    current_meaning = ""
    state = ""

    def flush() -> None:
        nonlocal current_phrase, current_glosses, current_meaning
        if current_phrase and current_glosses and current_meaning:
            entries.append(
                ChineseTeachingEntry(
                    phrase=current_phrase,
                    glosses=current_glosses[:],
                    meaning=current_meaning,
                )
            )
        current_phrase = ""
        current_glosses = []
        current_meaning = ""

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        phrase_match = re.match(r"^(?:\d+[.)]\s+)?([\u4e00-\u9fff]{1,8})$", line)
        if phrase_match:
            flush()
            current_phrase = phrase_match.group(1)
            state = ""
            continue
        if line.lower() == "meaning:":
            state = "gloss"
            continue
        if line.lower().startswith("whole phrase meaning and use"):
            state = "meaning"
            continue
        if line.startswith("-") and state == "gloss":
            gloss = line[1:].strip()
            gloss = strip_parenthetical_pinyin(gloss)
            current_glosses.append(gloss)
            continue
        if state == "meaning" and not current_meaning:
            current_meaning = line.strip()
            continue

    flush()
    return entries


def candidate_chinese_phrases(text: str, limit: int = 2) -> list[str]:
    phrases: list[str] = []
    for chunk in CHINESE_RUN_RE.findall(text):
        if not 2 <= len(chunk) <= 8:
            continue
        if chunk not in phrases:
            phrases.append(chunk)
        if len(phrases) >= limit:
            break
    return phrases


def extract_chinese_teaching_entries(text: str) -> list[ChineseTeachingEntry]:
    parsed_entries = parse_reformatted_teaching_entries(text)
    if parsed_entries:
        return parsed_entries

    phrases = candidate_chinese_phrases(text)
    if not phrases:
        return []

    payload = {
        "model": DEFAULT_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Create beginner Chinese teaching entries for the exact phrases provided by the user.\n"
                    "Return JSON only. No markdown.\n"
                    "Schema:\n"
                    "[{\"phrase\":\"你好\",\"glosses\":[\"你 You\",\"好 Good\"],\"meaning\":\"\\\"Hello.\\\" Used to greet one person.\"}]\n"
                    "Rules:\n"
                    "- Use exactly the phrases given by the user.\n"
                    "- `glosses` must be short English gloss lines, one per character or word.\n"
                    "- `meaning` must be English only.\n"
                    "- Keep the output compact and beginner-friendly.\n"
                    "- If a phrase is a greeting, the meaning should explain natural usage, not just literal meaning."
                ),
            },
            {
                "role": "user",
                "content": "\n".join(phrases),
            },
        ],
        "stream": False,
        "think": False,
    }
    response = create_response(payload)
    raw = extract_json_block(assistant_text(response["message"]))
    loaded = json.loads(raw)
    entries: list[ChineseTeachingEntry] = []
    if not isinstance(loaded, list):
        return entries
    for item in loaded:
        if not isinstance(item, dict):
            continue
        phrase = "".join(CHINESE_RUN_RE.findall(str(item.get("phrase", ""))))
        glosses_raw = item.get("glosses", [])
        meaning = str(item.get("meaning", "")).strip()
        if not phrase or not isinstance(glosses_raw, list) or not meaning:
            continue
        glosses = [str(gloss).strip() for gloss in glosses_raw if str(gloss).strip()]
        if not glosses:
            continue
        entries.append(ChineseTeachingEntry(phrase=phrase, glosses=glosses, meaning=meaning))
    return entries


def annotate_phrase_with_zhuyin(text: str) -> str:
    zhuyin = zhuyin_syllables(text)
    pinyin = pinyin_syllables(text)
    if not zhuyin:
        return text
    if pinyin:
        return f"{text} ({' '.join(zhuyin)}) ({' '.join(pinyin)})"
    return f"{text} ({' '.join(zhuyin)})"


def annotate_gloss_segment(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        chunk = match.group(0)
        syllables = zhuyin_syllables(chunk)
        if not syllables:
            return chunk
        if len(chunk) <= 4:
            return " ".join(f"{char} ({syllable})" for char, syllable in zip(chunk, syllables))
        return f"{chunk} ({' '.join(syllables)})"

    return CHINESE_RUN_RE.sub(replace, text)


def render_chinese_teaching_entry(entry: ChineseTeachingEntry) -> str:
    lines = [annotate_phrase_with_zhuyin(entry.phrase), "Meaning:"]
    for index, gloss in enumerate(entry.glosses):
        rendered_gloss = gloss
        if index < len(entry.phrase) and not text_contains_chinese(gloss):
            rendered_gloss = f"{entry.phrase[index]} {gloss}"
        lines.append(f"- {annotate_gloss_segment(rendered_gloss)}")
    lines.append(f"- {annotate_phrase_with_zhuyin(entry.phrase)} {entry.meaning}")
    return "\n".join(lines)


def normalize_chinese_teaching_line(line: str) -> str:
    stripped = strip_parenthetical_pinyin(line)
    if " - " not in stripped:
        chinese_only = "".join(CHINESE_RUN_RE.findall(stripped))
        if 1 <= len(chinese_only) <= 8:
            return annotate_phrase_with_zhuyin(chinese_only)
        return stripped

    parts = [part.strip() for part in stripped.split(" - ")]
    if not parts:
        return stripped

    prefix = ""
    head = parts[0]
    marker_match = re.match(r"^(\s*(?:[-*]|\d+[.)])\s+)(.+)$", head)
    if marker_match:
        prefix = marker_match.group(1)
        head = marker_match.group(2)

    phrase = "".join(CHINESE_RUN_RE.findall(head))
    if 1 <= len(phrase) <= 8:
        parts[0] = f"{prefix}{annotate_phrase_with_zhuyin(phrase)}"
    else:
        parts[0] = f"{prefix}{head}".strip()

    if len(parts) >= 2:
        parts[1] = annotate_gloss_segment(parts[1])

    return " - ".join(parts)


def annotate_chinese_with_zhuyin(text: str) -> str:
    if not text_contains_chinese(text):
        return text

    annotated_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        chinese_count = len(CHINESE_CHAR_RE.findall(line))

        if not chinese_count:
            annotated_lines.append(line)
            continue

        if chinese_count <= 24:
            annotated_lines.append(normalize_chinese_teaching_line(line))
            continue

        annotated_lines.append(line)

    return "\n".join(annotated_lines)


def postprocess_answer(answer: str, persona: str) -> str:
    if normalize_persona(persona) != "language_tutor":
        return answer
    if text_contains_chinese(answer):
        try:
            answer = reformat_chinese_teaching_response(answer)
            entries = extract_chinese_teaching_entries(answer)
            if entries:
                return "\n\n".join(render_chinese_teaching_entry(entry) for entry in entries)
        except Exception:
            pass
    return annotate_chinese_with_zhuyin(answer)


def run_agent_turn(
    user_text: str,
    tools: AgentTools,
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    messages.append({"role": "user", "content": user_text})
    include_tools = model_supports_tools(DEFAULT_MODEL)

    while True:
        payload = {
            "model": DEFAULT_MODEL,
            "messages": messages,
            "stream": False,
            "think": False,
        }
        if include_tools:
            payload["tools"] = tools.schemas()
        response = create_response(payload)
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
    configure_stdio()
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

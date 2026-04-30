import unittest
import uuid
from shutil import rmtree
from pathlib import Path

from agent import (
    DEFAULT_PERSONA,
    DEFAULT_TUTOR_MODE,
    LocalAgentSession,
    MemoryStore,
    PROMPTS_DIR,
    TUTOR_MODES,
    build_system_prompt,
    load_profile_seed,
    normalize_persona,
    normalize_tutor_mode,
)
from discord_bot import parse_allowed_channel_ids


class MemoryStoreTests(unittest.TestCase):
    def make_store_path(self) -> Path:
        base = Path(__file__).resolve().parent / ".tmp" / str(uuid.uuid4())
        base.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: rmtree(base, ignore_errors=True))
        return base / "memory.json"

    def test_add_and_list(self) -> None:
        store = MemoryStore(self.make_store_path())
        store.add("preference", "User likes Python.")

        items = store.list()

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["category"], "preference")
        self.assertEqual(items[0]["content"], "User likes Python.")

    def test_search(self) -> None:
        store = MemoryStore(self.make_store_path())
        store.add("project", "Build a Discord bot")
        store.add("decision", "Use SQLite for local state")

        results = store.search("sqlite")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["category"], "decision")

    def test_clear(self) -> None:
        path = self.make_store_path()
        store = MemoryStore(path)
        store.add("preference", "User likes Python.")

        store.clear()

        self.assertFalse(path.exists())
        self.assertEqual(store.list(), [])

    def test_persona_normalization_falls_back_to_default(self) -> None:
        self.assertEqual(normalize_persona("language tutor"), "language_tutor")
        self.assertEqual(normalize_persona("unknown"), DEFAULT_PERSONA)

    def test_tutor_mode_normalization_falls_back_to_default(self) -> None:
        self.assertEqual(normalize_tutor_mode("quiz"), "quiz")
        self.assertEqual(normalize_tutor_mode("unknown"), DEFAULT_TUTOR_MODE)

    def test_persona_prompt_changes_by_persona(self) -> None:
        tutor_prompt = build_system_prompt("language_tutor", "correction")
        memory_prompt = build_system_prompt("mnemosyne")

        self.assertIn("Language Tutor", tutor_prompt)
        self.assertIn("Current tutor mode: Correction (`correction`)", tutor_prompt)
        self.assertIn("Preserve the threads", memory_prompt)
        self.assertIn("Structured user profile seed", memory_prompt)

    def test_shared_memory_across_persona_sessions(self) -> None:
        path = self.make_store_path()
        mnemosyne = LocalAgentSession(path, persona="mnemosyne")
        tutor = LocalAgentSession(path, persona="language_tutor")

        mnemosyne.memory.add("preference", "User prefers Traditional Chinese explanations.")

        self.assertEqual(tutor.memory_count(), 1)
        self.assertEqual(tutor.memory.list()[0]["category"], "preference")

    def test_mnemosyne_prompt_file_exists(self) -> None:
        self.assertTrue((PROMPTS_DIR / "mnemosyne.md").exists())

    def test_language_tutor_prompt_file_exists(self) -> None:
        self.assertTrue((PROMPTS_DIR / "language_tutor.md").exists())

    def test_profile_seed_loads(self) -> None:
        profile = load_profile_seed()

        self.assertEqual(profile["identity"]["name"], "Cheng-Wei")
        self.assertIn("This file is a starting profile seed", profile["notes"][0])

    def test_tutor_mode_switch_updates_session(self) -> None:
        session = LocalAgentSession(self.make_store_path(), persona="language_tutor")

        session.switch_tutor_mode("quiz")

        self.assertEqual(session.tutor_mode, "quiz")
        self.assertEqual(session.tutor_mode_label(), TUTOR_MODES["quiz"].label)

    def test_parse_allowed_channel_ids(self) -> None:
        parsed = parse_allowed_channel_ids("123, 456, nope, 789")

        self.assertEqual(parsed, {123, 456, 789})


if __name__ == "__main__":
    unittest.main()

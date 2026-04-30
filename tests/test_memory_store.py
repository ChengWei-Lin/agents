import unittest
import uuid
from shutil import rmtree
from pathlib import Path

from agent import MemoryStore


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


if __name__ == "__main__":
    unittest.main()

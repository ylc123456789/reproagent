import unittest
from pathlib import Path

from reproagent.models import ReproTask


class TestModels(unittest.TestCase):
    def test_task_generates_id(self):
        task = ReproTask(paper_url="p", repo_url="r", workspace_dir=Path("runs/x"))
        self.assertTrue(task.task_id.startswith("repro-"))


if __name__ == "__main__":
    unittest.main()

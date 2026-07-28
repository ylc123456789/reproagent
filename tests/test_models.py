import unittest
from pathlib import Path

from reproagent.models import ReproTask


class TestModels(unittest.TestCase):
    def test_task_generates_id(self):
        task = ReproTask(paper_url="p", repo_url="r", workspace_dir=Path("runs/x"))
        self.assertTrue(task.task_id.startswith("repro-"))


if __name__ == "__main__":
    unittest.main()


def test_task_default_experiment_goal_is_empty(tmp_path):
    task = ReproTask(paper_url="paper", repo_url="repo", workspace_dir=tmp_path)

    assert task.experiment_goal == ""
    assert not task.confirm_before_experiment
    assert not task.plan_only

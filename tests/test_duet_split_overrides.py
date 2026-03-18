import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class DuetSplitOverrideTest(unittest.TestCase):
    def test_split_helper_honors_explicit_override(self) -> None:
        script = r'''
source scripts/duet/_splits.sh
export MERGE_POPULARITY_FORGET=1
export FORGET_SPLIT_OVERRIDE=city_forget_rare_1
export RETAIN_SPLIT_OVERRIDE=city_fast_retain_500
export FORGET_LABEL_OVERRIDE=city_forget_rare_1
set_forget_retain_splits
printf '%s\n' "${forget_retain_splits[@]}"
'''
        result = subprocess.run(
            ["bash", "-lc", script],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.stdout.strip(),
            "city_forget_rare_1 city_fast_retain_500 city_forget_rare_1",
        )


if __name__ == "__main__":
    unittest.main()

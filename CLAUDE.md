# Working rules for this repository

Correctness matters more than style. When changing this library:

- Work on a new branch with one logical fix per commit.
- Never change a test's expected value just to make it pass. If a test is wrong, prove it with an independent calculation.
- Add a regression test for every numerical bug you fix.
- Keep public function names and signatures unless I approve a change.
- Do not reformat code you are not fixing, so diffs stay readable.
- At the end, rerun all tests and notebooks and summarize what changed and what is still open.

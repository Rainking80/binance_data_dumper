# Workflow: Smart Commit
# Trigger: /commit

## Step 1: Quality Audit
- Run `ruff check .` and `pytest`.
- If either fails, stop and initiate the "Auto-Heal" workflow.

## Step 2: Contextual Message
- Analyze `git diff --cached` to understand the changes.
- Write a conventional commit message (e.g., `feat:`, `fix:`, `docs:`).

## Step 3: Execution
- Execute `git commit -m "[Generated Message]"`.
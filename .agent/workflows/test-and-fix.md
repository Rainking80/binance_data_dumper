# Workflow: Test & Auto-Fix
# Trigger: /test-fix

## Step 1: Discovery
- Identify all relevant test files in the `tests/` directory or files matching `test_*.py`.
- If no tests exist for the current file, generate a comprehensive pytest suite first.

## Step 2: Execution
- Run `pytest` using the terminal.
- Capture the full STDOUT and STDERR.

## Step 3: Analysis & Iteration
- IF tests pass: 
    - Report success and summarize coverage.
- IF tests fail:
    - Analyze the traceback to identify the root cause (Logic error, Missing Import, or Type Mismatch).
    - Propose a fix in a "Plan" artifact.
    - Upon user approval, apply the fix and **return to Step 2**.
    - Repeat up to 3 times before asking for manual intervention.
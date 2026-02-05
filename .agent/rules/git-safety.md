# Rule: Git Quality Gate

- **Pre-Commit Check:** Before performing any `git commit`, you MUST run `ruff check .` (or `flake8`).
- **Zero Tolerance:** If the linter reports any errors or warnings, you are forbidden from committing. 
- **Auto-Correction:** If linting fails, you must attempt to fix the issues using `ruff --fix` before re-running the check.
- **Reporting:** If you cannot fix an error automatically, you must present the error to the user and ask for guidance instead of bypassing the check.
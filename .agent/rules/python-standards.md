# Python Development Standards

## 1. Environment & Dependencies
- **Strictly use `uv` or `poetry`** for dependency management if detected in the workspace. Do not use raw `pip install` unless no lockfile exists.
- Before adding a new package, check `pyproject.toml` to see if a similar library is already present.

## 2. Code Structure & Style
- **Type Hinting:** All function signatures must include type hints (e.g., `def calculate(value: int) -> float:`).
- **Style Guide:** Follow PEP 8. Use double quotes for strings unless the string contains double quotes.
- **Docstrings:** Use Google-style docstrings for all public modules, classes, and functions.

## 3. Error Handling
- **No Silent Failures:** Never use bare `except: pass`. Always catch specific exceptions and log them using the `logging` module, not `print()`.
- **Validation:** Use `pydantic` for data validation if the project already uses it; otherwise, use standard `dataclasses`.

## 4. Testing Requirements
- **Pytest:** All new features must include a corresponding test in the `tests/` directory.
- **Mocking:** Use `unittest.mock` or `pytest-mock` for any network calls or disk I/O to ensure tests remain hermetic.

## 5. Antigravity Execution Rules
- **Safe Deletion:** You are prohibited from using `os.remove()` or `shutil.rmtree()` without a manual "User Approved" flag in the chat.
- **Refactoring:** Before a large refactor, generate a `PLAN.md` artifact outlining the changes and wait for my 'LGTM' or 'Proceed'.
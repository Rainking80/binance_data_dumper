# Rule: Real-time Documentation Sync

- **Mandatory Update:** For every new feature, API endpoint, or configuration variable added, you MUST update the `README.md` or the corresponding file in the `docs/` folder.
- **Auto-Summarization:** After finishing a task, compare the current state of the code with the existing `README.md`. If they are out of sync, propose a "Documentation Update" artifact.
- **Structure Requirements:**
    - **Installation:** Ensure any new dependencies are reflected in the setup instructions.
    - **Examples:** Provide a concise code snippet demonstrating how to use the new feature.
    - **Badges:** If a new test suite was added, ensure the "Tests" status in the README reflects the current local pass rate.
- **Tone:** Keep documentation technical, concise, and in line with the project's existing voice.
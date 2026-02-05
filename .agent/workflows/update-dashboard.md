# Workflow: Update Dashboard
# Trigger: /update-dash

## Step 1: Introspection
- Scan the `.agent/rules/` directory for all `.md` files.
- Scan the `.agent/workflows/` directory for all `.md` files.

## Step 2: Extraction
- Extract the "Trigger" and a brief description from each workflow.
- Summarize the core "Constraint" from each rule file.

## Step 3: File Update
- Create or overwrite `DASHBOARD.md` in the root directory.
- Use the following format:
    # 🎛️ Agent Control Panel
    ## 🚀 Available Commands
    | Command | Action |
    | :--- | :--- |
    | (List workflows here) |
    
    ## 🛡️ Active Rules
    - (List rules here)
    
    *Last Updated: [Current Date]*
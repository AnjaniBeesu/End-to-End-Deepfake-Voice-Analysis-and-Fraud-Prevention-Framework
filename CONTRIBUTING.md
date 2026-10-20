# Contributing to End-to-End Deepfake Voice Analysis Framework

Welcome **Nexus Spring of Code (NSoC) 2026** contributors! Follow these rules strictly to ensure your Pull Requests are reviewed, merged, and counted toward the leaderboard.

## 🚨 Mandatory NSoC 2026 Rules
1. **Wait for Assignment:** Do **NOT** open a Pull Request unless an issue has been officially assigned to you by the Project Admin (`@vanshi18s`). Unassigned PRs will be closed immediately.
2. **7-Day Activity Rule:** Once assigned an issue, you have **7 days** to show progress or submit a PR. Inactive assignees will be unassigned so others can work on the task.
3. **One Branch Per Issue:** Never work directly on `main` or share branches. Create a dedicated branch for each issue (`git checkout -b fix/issue-number`).
4. **Link Your Issue:** Your PR description **must** include `Closes #<issue-number>` (e.g., `Closes #12`).
5. **No Spam or AI-Generated PRs:** Bulk PRs, 0-line changes, and raw AI-generated code dumps will be rejected and reported.
6. **Respect `.gitignore`:** Never commit `venv/`, `node_modules/`, `.env` files, or large audio/model weights.

## 📊 Point System & Levels
Every valid issue and PR is tagged with `nsoc26` and one difficulty level by the Project Admin:
- **`level1` (3 Points):** Documentation, typo fixes, config tweaks, and minor adjustments.
- **`level2` (5 Points):** Next.js UI & frontend tasks (layouts, components, styling, audio visualizers, page-level changes).
- **`level3` (10 Points):** FastAPI backend core logic, database logging, audio preprocessing pipelines, and PyTorch/TensorFlow model integration.

## 🛠️ Step-by-Step Contribution Workflow
1. Fork the repository and clone it locally:

# AGENTS.md

## Scope

This project is for organizing code related to developing an RNN model of fuel moisture content.  

The project has separate related components: data retrieval and formatting for structured training data, validation procedure for forecast accuracy estimation, an operational predictor with the goal of real-time deployment, and then a lean python package that allows a user to install the core functionality and reuse in other contexts.

---

## File Edit Gate

* Do NOT modify any files by default.
* Wait for explicit user instruction before changing any file.
* Do NOT infer permission to edit files from requests to read, review, summarize, analyze, inspect, explain, extract, or respond to material.
* If the request is ambiguous, remain read-only and stop after reporting findings.
* Only treat a request as edit permission when the user clearly asks to create, modify, rewrite, patch, update, or delete specific files or project content.

---

## Workspace Rules

* Do NOT move files between repositories unless explicitly asked.
* Do NOT make cross-repository edits unless explicitly asked.
* Keep commits confined to the correct repository.
* Treat this top-level repository as shared context, not as the main location for research code or manuscript development.

---

## Shared Context

* Use this repository for shared Codex instructions and high-level project context.
* Keep durable project memory in explicit files that collaborators can inspect and edit.
* Prefer short, factual updates over long narrative notes.
* Record decisions and current priorities here when that context should be shared across repositories.

---

## Editing Style

* Be concise by default.
* Prefer short, direct wording over setup, repetition, or unnecessary framing.
* Make small, reviewable changes.
* When uncertain, ask or stop rather than guessing.

---

## Command Style

* When giving shell commands for work in this repository, prefer commands relative to the project root directory.
* Do NOT default to absolute paths when a project-root-relative command is sufficient.
* Assume commands are run from the repository root unless otherwise stated.
* Use absolute paths or `git -C` only when needed to avoid ambiguity across repositories or outside the project tree.
---

## Git Discipline

* Before any commit, check the actual repo state with `git status`.
* Before Codex commits, it must check whether the intended changes are staged or unstaged and give the user a WORKING COMMAND TO REVIEW THE ACTUAL DIFF in the correct repository before committing.
* Do not assume the current directory; use explicit paths or `git -C` when needed.
* Stage only the intended files; do not use broad staging commands unless verified safe.
* If the user asks to commit and the intended scope is clear, make the commit instead of only suggesting commands.

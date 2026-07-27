# F1Pulse Multi-Agent Orchestration Protocol

This project utilizes a 3-agent pipeline for high-performance C++20, Qt/QML, CMake, SQLite, and `llama.cpp` integration.

---

## ⚡ ACTIVE WORKFLOW & PROTOCOLS

1. **Full English Communication & Reasoning:**
   - All internal thinking, reasoning, code analysis, and conversational responses MUST be strictly in **English** to ensure optimal performance, low token usage, and clear terminal output.

2. **Superpowers Socratic Clarification Protocol:**
   - NEVER make blind assumptions if implementation details or requirements are ambiguous.
   - Ask the User targeted, high-leverage technical questions to remove key uncertainties BEFORE proposing code or architectural specifications.
   - Stop asking questions once requirements are crystal clear.

3. **Reproduction Over Guesswork:**
   - When fixing bugs or CMake compilation errors, request logs/stack traces or reproduce the error via minimal build commands first. Never guess root causes without diagnostic evidence.

4. **Approval & Protection Guardrails:**
   - **User (Lead) is the SOLE Approval Point** for all code changes.
   - **NO Git Operations:** Commands like `git commit`, `git push`, `git checkout` are STRICTLY FORBIDDEN for all agents.
   - **Protected Paths (READ-ONLY):** All agents are strictly prohibited from writing to or modifying files inside `.opencode/` and `third_party/llama.cpp/` (submodules).

---

## 🔄 Agent Pipeline Execution Flow

1. **Architecture & Strategy:** User + `@Architect` discuss high-level design, math, data flow, and trade-offs.
2. **Implementation:** `@Dev` clarifies technical details, proposes 2-3 implementation options with Pros/Cons, gets explicit approval from the User, and applies code changes.
3. **QA & Audit:** `@Reviewer` receives context from Architect and Dev, builds/tests the code, hunts for memory leaks, undefined behavior, and edge cases. If bugs are found, `@Reviewer` routes context back to `@Dev`.
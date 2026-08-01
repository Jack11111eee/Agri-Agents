# AGENTS.md

These rules apply only to the Agri-Agents project. They are based on the
reference rules in `/Users/huaxinzhang/Desktop/trifles/nba2kmobile/AGENTS.md`
and are intentionally limited to this project's needs.

## 1. Think Before Coding

- Read the project proposal documents and current `.gsd` artifacts before
  making implementation decisions.
- State assumptions when requirements, data sources, model behavior, or
  technology choices are uncertain.
- Keep the technical stack undecided until the planning workflow has researched
  the options and recorded the decision.
- Separate a research finding, a planning decision, and an implementation
  change. Do not present one as another.

## 2. Simplicity First

- Implement only the capability currently approved in the active plan.
- Do not add speculative model providers, databases, services, or deployment
  targets.
- Prefer established libraries and documented APIs over custom infrastructure.
- Keep the first milestone small enough to validate the core diagnostic loop.

## 3. Surgical Changes

- Touch only files required by the active task.
- Preserve existing user files and proposal documents unless the active plan
  explicitly calls for a revision.
- Do not edit global GSD/Pi configuration from this project.
- Do not alter unrelated files in the parent Git repository.
- Remove only imports, variables, or generated artifacts made obsolete by the
  current change.

## 4. Agent Collaboration

- Anthropic owns requirements, research, specifications, roadmaps, planning,
  plan checking, verification, audit, reflection, and replanning.
- OpenAI owns implementation, tests, debugging, code review, security review,
  build, lint, formatting, and implementation-time documentation.
- Every implementation task must leave objective verification evidence.
- Every review finding must be either fixed by an OpenAI execution agent or
  explicitly escalated to the user.
- Use no more than three parallel execution agents.

## 5. Human Approval Gates

The workflow must pause for explicit human confirmation at:

1. Requirements lock.
2. Execution plan approval.
3. Final acceptance.

Do not infer approval from silence, a timeout, a cancelled prompt, or an
ambiguous answer.

## 6. Domain and Safety

- Diagnosis and pesticide/control advice must retain evidence, provenance,
  version, and uncertainty.
- Do not expose API keys, personal contact data, farm-identifying data, or
  private conversation memory in logs, fixtures, prompts, or commits.
- Treat pesticide names, dosage, timing, and restrictions as safety-sensitive
  data requiring authoritative sources and explicit validation.
- The system must distinguish model suggestions from verified agricultural
  knowledge.

## 7. Verification

- No test, build, lint, or formatting command is assumed yet.
- When a command is established, record it in `.gsd/PREFERENCES.md` or the
  relevant planning artifact before relying on it as a gate.
- Prefer tests for behavior, static checks for contracts, and UAT for the
  user-visible diagnostic flow.
- When verification fails, preserve the failure evidence, fix the cause, and
  rerun the same check.

## 8. Git

- Automatic commits and pushes are allowed only for scoped project changes.
- Before committing, inspect `git diff --cached --name-only` and confirm every
  path belongs to `Agri-Agents`.
- Never use destructive Git commands to discard work.
- Do not clean, reset, or revert unrelated parent-repository changes.

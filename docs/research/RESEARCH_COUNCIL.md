# AI-Quant Research Council — multi-agent falsification protocol

Purpose: scientific falsification of AI-Quant research claims, not
consensus generation. Evidence outranks votes. Established 2026-09-15.

## Runtime (verified 2026-09-15, Claude Code 2.1.272)

- **Agent Teams: not available/enabled** in this environment (`ListAgents`
  shows no team roster; no team flags in settings). Subagents therefore
  **cannot message each other**; the main session (Lead) relays reports
  and counter-arguments faithfully and verbatim.
- Project reviewer agents live in `.claude/agents/` (loaded at session
  start; within the session that creates them, run the same role files
  through the read-only `Explore` agent type).
- The Lead is the main Claude Code session — there is no custom Lead agent.

## Roles

| Role | File | Tools | Default mode |
|---|---|---|---|
| Lead / Research Director | (main session) | all | defines the question, freezes variables, assigns blind reviews, relays debate, synthesizes, protects production; never advocates before reviews are in |
| Quant Red Team | `.claude/agents/quant-red-team.md` | Read, Grep, Glob, Bash (inspection) | falsify; FATAL / MATERIAL / MINOR / NO ISSUE |
| ML Training Researcher | `.claude/agents/ml-training-researcher.md` | same | strongest defensible case for model improvement; classify the binding limit |
| Data / Validation Auditor | `.claude/agents/data-validation-auditor.md` | same | does the experiment measure future predictive ability; evidence types; arbitrates method validity only |
| GPU / Systems Reviewer (optional) | `.claude/agents/gpu-systems-reviewer.md` | same | only for compute/throughput/optimization-semantics questions |

Default council = 3 specialists + Lead; the GPU reviewer only when the
question is computational; never more than 5 active roles without a
stated reason. No reviewer commits, pushes, merges, touches scheduled
tasks, production checkpoints, `my_holdings.csv`, latest reports, or
launches GPU work without Lead authorization.

## Protocol

**Round 1 — blind review.** Every relevant specialist gets the SAME
frozen research question, the SAME frozen evidence package (explicit file
list; in-flight results embargoed) and its own role file. No specialist
sees another's conclusions. Each returns exactly:
`CLAIM / EVIDENCE FOR / EVIDENCE AGAINST / BIGGEST CONFOUNDER / WHAT
WOULD FALSIFY MY VIEW / CONFIDENCE / RECOMMENDED NEXT EXPERIMENT`
plus its role table.

**Round 2 — adversarial debate.** The Lead relays all Round-1 reports to
every specialist (continuing their contexts). Each must challenge at
least one material claim of another with evidence ("find the strongest
reason this is wrong" — never "do you agree?"). The Red Team attacks the
ML Researcher's preferred interpretation; the ML Researcher defends,
narrows, concedes, or proposes a discriminating experiment; the Auditor
arbitrates method validity only.

**Round 3 — Lead synthesis.** `AGREED FINDINGS / FALSIFIED CLAIMS /
UNRESOLVED DISAGREEMENTS / EVIDENCE QUALITY / NEXT DISCRIMINATING
EXPERIMENT / PRODUCTION IMPLICATION`. Material dissent is preserved
verbatim; no fake consensus.

## Decision standard (no voting)

Each disputed conclusion gets an evidence state — `SUPPORTED /
WEAKLY_SUPPORTED / INCONCLUSIVE / WEAKLY_CONTRADICTED / FALSIFIED` — and
an evidence type — `MECHANICAL_CODE_PROOF / PAIRED_EXPERIMENT /
HISTORICAL_OOS / BURNED_DIAGNOSTIC / PROSPECTIVE / THEORETICAL_ONLY`.
One reviewer with a mechanical leakage proof or a demonstrated confound
overrides any majority. A burned backtest never becomes prospective
evidence by agreement. Claims must cite file paths, code behaviour,
experiment IDs, metrics, dates; "the other agents agree" is not evidence.

## Hypothesis tree

`reports/model_audit/v17/HYPOTHESIS_TREE.md` records every active
hypothesis: ID, claim, status (UNTESTED / TESTING / SUPPORTED /
WEAKLY_SUPPORTED / INCONCLUSIVE / REJECTED), supporting and
contradicting evidence, confounders, next discriminating experiment,
whether the evidence interval is burned, last updated. Failed
hypotheses are never deleted.

## Implementation separation & compute gate

Reviewers do not implement. After the Council recommends an experiment
the Lead creates a separate implementation task; a reviewer who did not
implement audits the implementation against the preregistered design
(designer ≠ implementer ≠ reviewer). Before any expensive GPU run the
Council agrees only on the DESIGN (hypothesis, variables, controls,
metrics, thresholds, evaluation dates, seed IDs, compute budget, stop
criteria) — never on the expected outcome — and a Red-Team FATAL
finding blocks the launch until resolved.

## Session log

- **Session 01 — 2026-09-15 — long training (100+ epochs).** Synthesis: `reports/model_audit/v17/COUNCIL_01_long_training_synthesis.md`. Hypothesis updated: H-EPOCH-LONG (TESTING → WEAKLY_CONTRADICTED on mechanism arithmetic, OOS results still embargoed). Process lessons: (1) the harness does not persist subagent output files, so the Lead must archive each Round-1 report to disk before relaying it in Round 2; (2) Round 2 via `SendMessage` to the same agent IDs preserves each member's context; (3) every member had at least one headline claim falsified on recomputation and conceded it, which is the intended behaviour of the adversarial round; (4) preregistrations must be committed before launch to be git-verifiable.
- **Session 02 — 2026-09-21 — P4-B0-LONG post-unblinding review.** Synthesis: `reports/model_audit/v17/COUNCIL_02_long_training_unblinded_synthesis.md`. Unblinding followed a preregistered amendment committed before any artifact was decoded (`c6e6a6d`), a committed evaluator (`7d5378b`), and a mechanism report committed before OOS (`93fff20`). H-EPOCH-LONG recorded as REJECTED (frozen recipe only), met at minimum margins and not significant; new entries H-RESIDUAL-SIGNAL, H-FACTOR-PREMIUM, H-LR-SCHEDULE (early), H-WEIGHT-AVERAGING. Session-01 finding A8 falsified (proxy error; erratum appended to the Session-01 synthesis). Process lessons: (1) condition blind predictions on values already known, and state them as intervals; (2) check mechanism stories against logged timing before calling them mechanical facts; (3) verify proxies before an agreed finding rests on them; (4) replace default-outcome rules with equivalence bounds; (5) make blinding mechanical by storing outcome channels separately; (6) native project agents now load from `.claude/agents/`, and Round 2 via `SendMessage` preserves each member's context.

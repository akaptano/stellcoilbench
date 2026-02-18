#!/usr/bin/env python3
"""
LLM-powered endpoints for the StellCoilBench Knowledge Base.

Provides two main functions used by the KB server's POST /brief and POST /propose
routes:

- **call_brief**: Generates a 2–4 paragraph research brief from recent optimization
  runs, failure statistics, and optionally relevant paper excerpts. Summarizes
  what's working, common failure modes, and promising directions.

- **call_propose**: Generates a batch of mutation/exploration actions (JSON) for
  the CI autopilot. Uses top parent runs, failure stats, and policy constraints
  to propose new cases.

Both require the LLM to be configured via KB_LLM_* environment variables
(see knowledge.llm.llm_client).
"""
from __future__ import annotations

import json
from typing import Any


BRIEF_SYSTEM = """You are a research assistant for StellCoilBench, a stellarator coil optimization benchmark.
Given context from recent optimization runs and optionally from papers, produce a concise research brief (2-4 paragraphs).
Focus on: what's working, common failure modes, promising directions, and any insights from the literature.
Cite runs by case_id and papers by title when relevant."""

PROPOSE_SYSTEM = """You are an expert at proposing stellarator coil optimization cases for StellCoilBench.
Given context (top runs, failure stats, policy constraints), output a JSON array of mutation actions.
Each action is either:
- {"type": "mutate", "parent_id": "<case_id>", "overrides": {"ncoils": 5, "cc_threshold": 1.2, "surface": "...", ...}}
- {"type": "explore", "surface": "...", "ncoils": 4, "order": 8, "thresholds": {"cc_threshold": 1.0, "cs_threshold": 2.0, "curvature_threshold": 2.0, "msc_threshold": 2.0, "length_threshold": 150}}

Rules:
- Only use surfaces, ncoils, order from the policy's allowed lists.
- For mutate: parent_id must be from top_parents. Overrides are optional (ncoils, order, surface, cc_threshold, cs_threshold, curvature_threshold, msc_threshold, length_threshold, force_threshold, torque_threshold).
- For explore: surface, ncoils, order required; thresholds optional (use policy ranges).
- Output ONLY valid JSON array, no markdown or explanation."""


def _format_runs_for_brief(runs: list[dict]) -> str:
    """Format top runs as a compact text block for the brief prompt."""
    lines = []
    for r in runs[:10]:
        cid = r.get("case_id", "?")
        success = r.get("success", False)
        score = r.get("total_score", "?")
        cfg = r.get("case_config", {})
        sp = cfg.get("surface_params", {})
        surface = sp.get("surface", "?") if isinstance(sp, dict) else "?"
        cp = cfg.get("coils_params", {})
        ncoils = cp.get("ncoils", "?") if isinstance(cp, dict) else "?"
        order = cp.get("order", "?") if isinstance(cp, dict) else "?"
        status = "SUCCESS" if success else "FAILED"
        lines.append(f"- {cid}: {status} score={score} | {surface} ncoils={ncoils} order={order}")
    return "\n".join(lines) if lines else "(no runs)"


def _format_stats_for_brief(stats: dict) -> str:
    """Format failure statistics as a compact string for the brief prompt."""
    parts = []
    if stats.get("total"):
        parts.append(f"Recent runs: {stats['total']}, fail_rate={stats.get('fail_rate', 0):.2f}")
    if stats.get("failure_classes"):
        parts.append(f"Failure classes: {json.dumps(stats['failure_classes'])}")
    if stats.get("failure_reasons"):
        parts.append(f"Failure reasons: {json.dumps(stats['failure_reasons'])}")
    return "; ".join(parts) if parts else "(no stats)"


def call_brief(
    runs: list[dict],
    stats: dict,
    papers: list[dict],
    *,
    query: str = "",
    context: str = "general",
) -> dict[str, Any]:
    """Generate a research brief from runs, stats, and paper excerpts.

    Parameters
    ----------
    runs : list[dict]
        Top optimization run summaries (case_id, success, total_score, case_config).
    stats : dict
        Failure statistics (total, fail_rate, failure_classes, failure_reasons).
    papers : list[dict]
        Relevant paper chunks (title, text/chunk_text).
    query : str, optional
        Optional user query to focus the brief.
    context : str, optional
        Context label (e.g. "general", "exploit") for the prompt.

    Returns
    -------
    dict
        {"brief": str, "citations": list} or {"error": str, "brief": "", "citations": []}
        if LLM is not available or not configured.
    """
    try:
        from knowledge.llm.llm_client import complete, is_available
    except ImportError:
        return {"error": "LLM not available", "brief": "", "citations": []}

    if not is_available():
        return {"error": "LLM not configured (set KB_LLM_* env vars)", "brief": "", "citations": []}

    runs_text = _format_runs_for_brief(runs)
    stats_text = _format_stats_for_brief(stats)
    papers_text = ""
    citations = []
    for p in papers[:5]:
        title = p.get("title", p.get("paper_id", "?"))
        text = p.get("text", p.get("chunk_text", ""))[:500]
        papers_text += f"\n[{title}]: {text}...\n"
        citations.append({"type": "paper", "title": title})

    for r in runs[:5]:
        citations.append({"type": "run", "case_id": r.get("case_id", "?")})

    user_content = f"""Context: {context}
Query: {query or '(general overview)'}

Top runs:
{runs_text}

Recent stats: {stats_text}
"""
    if papers_text:
        user_content += f"\nRelevant paper excerpts:\n{papers_text}"

    user_content += "\nProduce a concise research brief (2-4 paragraphs)."

    messages = [
        {"role": "system", "content": BRIEF_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    brief = complete(messages, max_tokens=1500, temperature=0.5)
    return {"brief": brief, "citations": citations}


def call_propose(
    context: dict,
    policy: dict,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Generate mutation/exploration actions for the next CI batch.

    Parameters
    ----------
    context : dict
        Build context with top_parents, failure_stats, recent_config_hashes.
    policy : dict
        Proposer policy (exploration surfaces, ncoils, order, mutation ranges).
    batch_size : int, optional
        Number of actions to propose (default 8).

    Returns
    -------
    dict
        {"actions": list[dict]} or {"error": str, "actions": []} if LLM fails.
        Each action is {"type": "mutate"|"explore", ...} with overrides/surface/etc.
    """
    try:
        from knowledge.llm.llm_client import complete_json, is_available
    except ImportError:
        return {"error": "LLM not available", "actions": []}

    if not is_available():
        return {"error": "LLM not configured (set KB_LLM_* env vars)", "actions": []}

    top_parents = context.get("top_parents", [])
    failure_stats = context.get("failure_stats", {})

    # Build allowed values from policy
    expl = policy.get("exploration", {})
    surfaces = expl.get("surfaces", ["input.LandremanPaul2021_QA"])
    ncoils_choices = expl.get("ncoils_choices", [3, 4, 5, 6, 7])
    order_choices = expl.get("order_choices", [4, 6, 8])
    mut = policy.get("mutation", {})
    if not ncoils_choices:
        ncoils_choices = mut.get("ncoils_choices", [3, 4, 5, 6, 7])
    if not order_choices:
        order_choices = mut.get("order_choices", [4, 6, 8])

    parent_summaries = []
    for p in top_parents[:10]:
        cid = p.get("case_id", "?")
        score = p.get("total_score", "?")
        cfg = p.get("case_config", {})
        sp = cfg.get("surface_params", {})
        surface = sp.get("surface", "?") if isinstance(sp, dict) else "?"
        cp = cfg.get("coils_params", {})
        ncoils = cp.get("ncoils", "?") if isinstance(cp, dict) else "?"
        order = cp.get("order", "?") if isinstance(cp, dict) else "?"
        parent_summaries.append(f"  - {cid}: score={score}, surface={surface}, ncoils={ncoils}, order={order}")

    user_content = f"""Propose {batch_size} cases for the next optimization batch.

Policy constraints:
- Allowed surfaces: {surfaces}
- Allowed ncoils: {ncoils_choices}
- Allowed order: {order_choices}

Top parent runs (for mutate):
{chr(10).join(parent_summaries) if parent_summaries else '  (none)'}

Failure stats: fail_rate={failure_stats.get('fail_rate', 0):.2f}, failure_classes={failure_stats.get('failure_classes', {})}

Output a JSON array of exactly {batch_size} actions. Mix mutate and explore. For mutate use parent_id from the list above.
Output ONLY the JSON array, no other text."""

    messages = [
        {"role": "system", "content": PROPOSE_SYSTEM},
        {"role": "user", "content": user_content},
    ]

    try:
        result = complete_json(messages, max_tokens=4096)
        if isinstance(result, list):
            actions = result
        elif isinstance(result, dict) and "actions" in result:
            actions = result["actions"]
        else:
            actions = []
        return {"actions": actions}
    except (json.JSONDecodeError, TypeError) as e:
        return {"error": str(e), "actions": []}

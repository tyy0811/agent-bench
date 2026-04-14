"""Deterministic and LLM-judge evaluation metrics."""

from __future__ import annotations

import json
import re

import structlog

from agent_bench.agents.orchestrator import AgentResponse
from agent_bench.core.provider import LLMProvider
from agent_bench.core.types import Message, Role

logger = structlog.get_logger()


# --- Deterministic metrics (free, CI-safe) ---


def retrieval_precision_at_k(
    retrieved_sources: list[str], expected_sources: list[str], k: int = 5
) -> float:
    """Fraction of retrieved sources (up to k) that are in the expected set."""
    if not retrieved_sources:
        return 0.0
    top_k = retrieved_sources[:k]
    expected = set(expected_sources)
    hits = sum(1 for s in top_k if s in expected)
    return hits / len(top_k)


def retrieval_recall_at_k(
    retrieved_sources: list[str], expected_sources: list[str], k: int = 5
) -> float:
    """Fraction of expected sources that appear in the retrieved set (up to k)."""
    if not expected_sources:
        return 0.0
    top_k = set(retrieved_sources[:k])
    return sum(1 for s in expected_sources if s in top_k) / len(expected_sources)


def keyword_hit_rate(answer: str, expected_keywords: list[str]) -> float:
    """Fraction of expected keywords that appear (case-insensitive) in the answer."""
    if not expected_keywords:
        return 0.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return hits / len(expected_keywords)


def source_presence(response: AgentResponse) -> bool:
    """Does the response contain at least one source reference?"""
    return len(response.sources) > 0


def grounded_refusal(answer: str, category: str) -> bool:
    """For out_of_scope: does the answer correctly refuse AND cite no sources?

    "Cite no sources" means no [source: X.md] citations appear in the answer
    text, not that retrieval returned zero candidates. On any non-trivial
    out-of-scope query, retrieval will still return low-relevance candidates
    (unless the grounded-refusal gate fires at the tool level, which only
    catches the thinnest queries). The agent is expected to inspect the
    candidates, find nothing relevant, and refuse without citing anything —
    and that refusal shape is what this metric measures.

    Returns True if:
    - Category is not out_of_scope (metric not applicable)
    - Category is out_of_scope AND answer contains refusal language AND the
      answer text contains no [source: ...] citations
    """
    if category != "out_of_scope":
        return True  # not applicable
    refusal_phrases = [
        "does not contain",
        "no information",
        "not contain",
        "not available",
        "not found",
        "cannot find",
        "no relevant",
        "outside the scope",
    ]
    answer_lower = answer.lower()
    has_phrase_refusal = any(phrase in answer_lower for phrase in refusal_phrases)
    # Canonical shape taught by the system prompt at core/prompts.py:17-18:
    # "not in the {corpus_label} documentation". Narrow regex anchors on
    # "documentation" within 60 chars so plain "not in the" fragments from
    # retrieval answers ("not in the same scope", "not in the default range")
    # do not count as refusals.
    has_canonical_refusal = bool(
        re.search(r"\bnot in the\b[^.]{0,60}\bdocumentation\b", answer, re.IGNORECASE)
    )
    has_refusal = has_phrase_refusal or has_canonical_refusal
    cites_in_answer = re.findall(r"\[source:\s*[^\]]+\]", answer, re.IGNORECASE)
    return has_refusal and len(cites_in_answer) == 0


def citation_accuracy(answer: str, sources: list[str]) -> float:
    """Fraction of [source: X.md] inline citations that match the structured sources list.

    Catches hallucinated citations (LLM cites a file that wasn't retrieved).
    """
    pattern = r"\[source:\s*(.+?)\]"
    cited = re.findall(pattern, answer)
    if not cited:
        return 1.0  # no citations to check
    source_set = set(sources)
    hits = sum(1 for c in cited if c.strip() in source_set)
    return hits / len(cited)


def tool_call_count(response: AgentResponse) -> int:
    """Total number of tool calls made."""
    return len(response.tools_used)


def calculator_used_when_expected(
    response: AgentResponse,
    requires_calculator: bool,
) -> bool:
    """Did the agent use the calculator tool when the question required it?"""
    if not requires_calculator:
        return True  # not applicable
    return "calculator" in response.tools_used


# --- LLM-judge metrics (costs money, manual) ---

_FAITHFULNESS_PROMPT = """\
You are evaluating whether an AI assistant's answer \
is fully supported by the provided source passages.

Source passages:
{chunks}

Answer to evaluate:
{answer}

Score the answer's faithfulness to the sources from 0.0 to 1.0:
- 1.0: Every claim is directly supported by the sources
- 0.5: Some claims are supported, others are extrapolated
- 0.0: The answer contradicts or is entirely unsupported

Respond with ONLY a JSON object:
{{"score": 0.8, "reasoning": "brief explanation"}}"""

_CORRECTNESS_PROMPT = """\
You are evaluating whether an AI assistant's answer \
is factually correct compared to a reference answer.

Reference answer:
{reference}

Answer to evaluate:
{answer}

Score correctness from 0.0 to 1.0:
- 1.0: All key facts match the reference
- 0.5: Some facts are correct, some are missing or wrong
- 0.0: The answer is factually incorrect

Respond with ONLY a JSON object:
{{"score": 0.8, "reasoning": "brief explanation"}}"""


async def answer_faithfulness(
    answer: str,
    source_chunks: list[str],
    judge_provider: LLMProvider,
) -> float | None:
    """LLM-judged: is the answer supported by the sources? 0.0-1.0."""
    chunks_text = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(source_chunks))
    prompt = _FAITHFULNESS_PROMPT.format(chunks=chunks_text, answer=answer)

    return await _judge_call(prompt, judge_provider)


async def answer_correctness(
    answer: str,
    reference_answer: str,
    judge_provider: LLMProvider,
) -> float | None:
    """LLM-judged: is the answer factually correct vs reference? 0.0-1.0."""
    prompt = _CORRECTNESS_PROMPT.format(reference=reference_answer, answer=answer)

    return await _judge_call(prompt, judge_provider)


async def _judge_call(prompt: str, provider: LLMProvider) -> float | None:
    """Make a judge call and parse the JSON response."""
    try:
        response = await provider.complete(
            [Message(role=Role.USER, content=prompt)],
            temperature=0.0,
            max_tokens=256,
        )
        data = json.loads(response.content)
        score = float(data["score"])
        reasoning = data.get("reasoning", "")
        logger.info("llm_judge_result", score=score, reasoning=reasoning)
        return max(0.0, min(1.0, score))
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        logger.warning("llm_judge_parse_error", error=str(e), raw=response.content[:200])
        return None
    except Exception as e:
        logger.error("llm_judge_call_error", error=str(e))
        return None

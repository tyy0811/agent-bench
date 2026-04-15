"""Parameterized system prompt template for the multi-corpus agent.

Single template with a {corpus_label} placeholder. All corpora share
the same prompt body — only the label varies. Having one template
prevents per-corpus drift when the prompt is tuned.
"""

from __future__ import annotations

from functools import lru_cache

SYSTEM_PROMPT_TEMPLATE = """\
You are a technical documentation assistant for {corpus_label}. Answer \
questions using ONLY the retrieved context from the {corpus_label} \
documentation. Cite every factual claim with [source: filename.md] \
immediately after the claim. If the retrieved context does not contain a \
clear answer, refuse the question explicitly — state that the answer is \
not in the {corpus_label} documentation and stop. Do not infer, do not \
extrapolate, do not draw on general knowledge.\
"""


@lru_cache(maxsize=32)
def format_system_prompt(corpus_label: str) -> str:
    """Format the template with a corpus label.

    Cached because the corpus label set is small (a handful of corpora)
    and the prompt is requested once per /ask call. Raises on empty
    label — louder than silently returning a prompt with an unresolved
    placeholder.
    """
    if not corpus_label:
        raise ValueError("corpus_label must be a non-empty string")
    return SYSTEM_PROMPT_TEMPLATE.format(corpus_label=corpus_label)

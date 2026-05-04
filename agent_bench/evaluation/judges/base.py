"""Judge ABC, ScoreResult, Rubric, MockJudge, abstain-reason constants.

The Judge layer supersedes the continuous-scale answer_faithfulness /
answer_correctness functions in agent_bench/evaluation/metrics.py. See
docs/plans/2026-05-04-judge-layer-v1-design.md for the supersession
rationale and the six-axis comparison table.
"""

from __future__ import annotations

import hashlib
import json as _json
import random
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

import structlog
import yaml
from pydantic import BaseModel, Field

from agent_bench.core.provider import (
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from agent_bench.core.types import Message, Role

if TYPE_CHECKING:
    from agent_bench.agents.orchestrator import AgentResponse
    from agent_bench.core.provider import LLMProvider
    from agent_bench.evaluation.harness import GoldenQuestion

logger = structlog.get_logger()

# --- Abstain-reason constants ---
#
# Failure-as-abstain ScoreResults carry a reasoning string with one of
# these prefixes. The calibration report pattern-matches against these
# constants for the four-way breakdown in the >20% abstain-rate flag.
# Genuine model abstain (rubric-allowed) uses the empty-string sentinel.

ABSTAIN_REASON_PROVIDER_EXHAUSTED = "judge_call_failed_after_retry: "
ABSTAIN_REASON_SCHEMA_PARSE = "schema_parse_failed_after_retry: "
ABSTAIN_REASON_OUT_OF_RANGE = "score_out_of_range_after_retry: "
ABSTAIN_REASON_GENUINE = ""


class ScoreResult(BaseModel):
    """One judge call's result. Self-contained provenance — no run
    metadata cross-reference needed for κ aggregation.

    Field order matters: reasoning + evidence_quotes come BEFORE score
    in both Pydantic field order and the JSON schema sent to the model,
    so the score conditions on the reasoning rather than being
    post-hoc rationalized.
    """

    # Reasoning-first ordering — load-bearing for the JSON schema
    reasoning: str
    evidence_quotes: list[str] = Field(default_factory=list)
    score: int | Literal["Unknown"]

    # Provenance
    judge_id: str
    rubric_version: str
    prompt_seed: int = 0
    system_output_hash: str

    # Operations
    cost_usd: float
    latency_ms: float

    @property
    def abstained(self) -> bool:
        return self.score == "Unknown"


class RubricLevel(BaseModel):
    """One score level in a rubric, with anchored examples.

    Parsed from markdown sections under `## Score N` headers. The
    `examples` list contains the H3 sub-sections (`### Example X`)
    each with a thinking-trace explanation of why that output got
    that score.
    """

    score: int
    description: str
    examples: list[str]  # raw markdown of `### Example` sections


class Rubric(BaseModel):
    """A scoring rubric loaded from a markdown file with YAML frontmatter.

    Construction validates aggressively: scale ∈ {binary, three_point},
    levels arity matches scale, every level has at least one anchored
    example. ValidationError raises with file path + field path so a
    Day-1 rubric typo doesn't surface as a Day-2 judge.score crash with
    API budget already spent.
    """

    dimension: Literal[
        "groundedness", "relevance", "completeness", "citation_faithfulness"
    ]
    scale: Literal["binary", "three_point"]
    reference_based: bool
    abstain_allowed: bool
    levels: list[RubricLevel]
    body_markdown: str

    @property
    def source_hash(self) -> str:
        """SHA-256 of the canonical body. Immutable per file content,
        independent of git state. Used as ScoreResult.rubric_version.
        """
        return hashlib.sha256(self.body_markdown.encode("utf-8")).hexdigest()

    @classmethod
    def from_markdown_file(cls, path: Path | str) -> Self:
        path = Path(path)
        body = path.read_text(encoding="utf-8")

        # Parse YAML frontmatter delimited by --- ... ---
        fm_match = re.match(r"^---\n(.+?)\n---\n(.*)$", body, re.DOTALL)
        if not fm_match:
            raise ValueError(
                f"Rubric {path.name}: missing YAML frontmatter "
                f"(expected --- ... --- block at top of file)"
            )
        try:
            frontmatter = yaml.safe_load(fm_match.group(1)) or {}
        except yaml.YAMLError as e:
            raise ValueError(
                f"Rubric {path.name}: frontmatter YAML parse error: {e}"
            ) from e

        required = {"dimension", "scale", "reference_based", "abstain_allowed"}
        missing = required - frontmatter.keys()
        if missing:
            raise ValueError(
                f"Rubric {path.name}: frontmatter missing fields: {sorted(missing)}"
            )

        scale = frontmatter["scale"]
        if scale not in ("binary", "three_point"):
            raise ValueError(
                f"Rubric {path.name}: invalid scale {scale!r}; "
                f"must be 'binary' or 'three_point'"
            )

        # Parse levels by ## Score N headers
        body_no_fm = fm_match.group(2)
        level_pattern = re.compile(
            r"^## Score (\d+)\n(.*?)(?=^## Score |\Z)", re.MULTILINE | re.DOTALL
        )
        raw_levels: list[tuple[int, str]] = [
            (int(m.group(1)), m.group(2)) for m in level_pattern.finditer(body_no_fm)
        ]

        expected_arity = 2 if scale == "binary" else 3
        if len(raw_levels) != expected_arity:
            raise ValueError(
                f"Rubric {path.name}: arity mismatch — scale {scale!r} "
                f"requires {expected_arity} levels, found {len(raw_levels)}"
            )

        # Parse examples (### Example) per level
        levels: list[RubricLevel] = []
        for score, level_body in raw_levels:
            example_pattern = re.compile(
                r"^### (Example .+?)\n(.*?)(?=^### |\Z)", re.MULTILINE | re.DOTALL
            )
            examples = [m.group(0) for m in example_pattern.finditer(level_body)]
            if not examples:
                raise ValueError(
                    f"Rubric {path.name}: level Score {score} has no "
                    f"anchored example (expected at least one ### Example header)"
                )
            description = level_body.split("###", 1)[0].strip()
            levels.append(
                RubricLevel(score=score, description=description, examples=examples)
            )

        return cls(
            dimension=frontmatter["dimension"],
            scale=scale,
            reference_based=bool(frontmatter["reference_based"]),
            abstain_allowed=bool(frontmatter["abstain_allowed"]),
            levels=levels,
            body_markdown=body,
        )

    def render_prompt(self, *, level_permutation_seed: int = 0) -> str:
        """Render the rubric body for inclusion in a judge prompt.

        If level_permutation_seed > 0, levels are reordered deterministically
        using a seeded PRNG. seed=0 returns the canonical order.
        """
        if level_permutation_seed == 0:
            return self.body_markdown
        rng = random.Random(level_permutation_seed)
        permuted_levels = list(self.levels)
        rng.shuffle(permuted_levels)
        # Reconstruct: keep frontmatter + intro paragraphs intact;
        # reorder the ## Score N sections.
        fm_match = re.match(r"^(---\n.+?\n---\n)(.*)$", self.body_markdown, re.DOTALL)
        if not fm_match:
            return self.body_markdown  # defensive — should never happen post-construction
        head = fm_match.group(1)
        rest = fm_match.group(2)
        intro = re.split(r"^## Score ", rest, maxsplit=1, flags=re.MULTILINE)[0]
        permuted_body = head + intro + "\n".join(
            f"## Score {lvl.score}\n{lvl.description}\n" + "\n".join(lvl.examples)
            for lvl in permuted_levels
        )
        return permuted_body


class Judge(ABC):
    """Per-dimension LLM judge. Concrete subclasses implement score()
    for one rubric dimension; they are thin (~30 lines) and not
    factored against a shared base method (see design doc for why).
    """

    def __init__(
        self,
        judge_provider: "LLMProvider",
        rubric: Rubric,
        model_id: str,
    ) -> None:
        self.judge_provider = judge_provider
        self.rubric = rubric
        self.model_id = model_id
        self.judge_id = f"{model_id}_{rubric.dimension}"

    @abstractmethod
    async def score(
        self,
        item: "GoldenQuestion",
        output: "AgentResponse",
        *,
        prompt_seed: int = 0,
    ) -> ScoreResult:
        """Score one (item, output) pair against this judge's rubric.

        Returns a ScoreResult whose system_output_hash is computed from
        (item.id, output.answer, sorted(output.sources)). Failures map
        to abstain via the abstain-reason constants; provider non-
        retryable errors raise (caller bug, not noise).
        """
        ...


class MockJudge(Judge):
    """Pre-baked-verdict judge for deterministic tests. No API calls.

    Constructor takes verdicts: dict[item_id, ScoreResult]. score()
    raises LookupError on missing keys — never returns a default —
    so test fixtures are self-checking. A separate fixture-validation
    test (test_mockjudge_coverage.py) walks item.id across all goldens
    and asserts every MockJudge instance has coverage for the items
    its tests reference.

    Mirrors the MockProvider pattern at agent_bench/core/provider.py.
    """

    def __init__(self, verdicts: dict[str, ScoreResult]) -> None:
        # MockJudge does not need provider/rubric/model_id; supply
        # placeholder values so the ABC's __init__ doesn't matter.
        self.judge_provider = None  # type: ignore[assignment]
        self.rubric = None  # type: ignore[assignment]
        self.model_id = "mock"
        self.judge_id = "mock_judge"
        self._verdicts = verdicts

    async def score(
        self,
        item: "GoldenQuestion",
        output: "AgentResponse",
        *,
        prompt_seed: int = 0,
    ) -> ScoreResult:
        if item.id not in self._verdicts:
            raise LookupError(
                f"MockJudge has no pre-baked verdict for item_id {item.id!r}; "
                f"available: {sorted(self._verdicts.keys())[:5]}"
                + (" ..." if len(self._verdicts) > 5 else "")
            )
        return self._verdicts[item.id]


# --- _call_judge_with_retry helper ---

_STRICT_REPROMPT_SUFFIX = (
    "\n\nSTRICT FORMATTING NOTE: respond ONLY with a JSON object matching "
    "the schema; reasoning first, then evidence_quotes, then score."
)


async def _call_judge_with_retry(
    *,
    provider: "LLMProvider",
    prompt: str,
    valid_scores: set[int],
    judge_id: str,
    rubric_version: str,
    prompt_seed: int,
    system_output_hash: str,
    item_id: str,
    abstain_allowed: bool = True,
    max_tokens: int = 512,
) -> ScoreResult:
    """Send prompt to provider; one retry with strict reprompt on
    schema-parse / score-out-of-range; abstain on persistent failure
    or provider exhaustion. Re-raises unknown exceptions (caller bugs).
    """
    accumulated_cost = 0.0
    accumulated_latency = 0.0

    for attempt in range(2):  # 2 = original + one retry
        send_prompt = prompt if attempt == 0 else prompt + _STRICT_REPROMPT_SUFFIX
        start = time.perf_counter()
        try:
            response = await provider.complete(
                [Message(role=Role.USER, content=send_prompt)],
                temperature=0.0,
                max_tokens=max_tokens,
            )
        except (ProviderRateLimitError, ProviderTimeoutError) as e:
            return ScoreResult(
                reasoning=f"{ABSTAIN_REASON_PROVIDER_EXHAUSTED}{type(e).__name__}: {e}",
                evidence_quotes=[],
                score="Unknown",
                judge_id=judge_id,
                rubric_version=rubric_version,
                prompt_seed=prompt_seed,
                system_output_hash=system_output_hash,
                cost_usd=accumulated_cost,
                latency_ms=accumulated_latency + (time.perf_counter() - start) * 1000,
            )
        # Other exceptions (caller bugs like 401, 400) propagate.
        accumulated_cost += response.usage.estimated_cost_usd
        accumulated_latency += (time.perf_counter() - start) * 1000
        last_raw = response.content[:300]

        # Parse
        try:
            data = _json.loads(response.content)
            reasoning = str(data["reasoning"])
            evidence_quotes = list(data.get("evidence_quotes", []))
            raw_score = data["score"]
        except (_json.JSONDecodeError, KeyError, TypeError) as e:
            cause = ABSTAIN_REASON_SCHEMA_PARSE
            if attempt == 0:
                logger.warning(
                    "judge_first_attempt_failure",
                    judge_id=judge_id,
                    item_id=item_id,
                    provider=type(provider).__name__,
                    failure_cause=cause,
                    attempt_index=1,
                )
                continue
            return ScoreResult(
                reasoning=f"{cause}raw={last_raw!r} parse_error={e}",
                evidence_quotes=[],
                score="Unknown",
                judge_id=judge_id,
                rubric_version=rubric_version,
                prompt_seed=prompt_seed,
                system_output_hash=system_output_hash,
                cost_usd=accumulated_cost,
                latency_ms=accumulated_latency,
            )

        # Score validation
        if raw_score == "Unknown":
            if not abstain_allowed:
                cause = ABSTAIN_REASON_OUT_OF_RANGE
                if attempt == 0:
                    logger.warning(
                        "judge_first_attempt_failure",
                        judge_id=judge_id,
                        item_id=item_id,
                        provider=type(provider).__name__,
                        failure_cause=cause,
                        attempt_index=1,
                    )
                    continue
                return ScoreResult(
                    reasoning=(
                        f"{cause}model returned 'Unknown' but rubric "
                        f"abstain_allowed=False"
                    ),
                    evidence_quotes=[],
                    score="Unknown",
                    judge_id=judge_id,
                    rubric_version=rubric_version,
                    prompt_seed=prompt_seed,
                    system_output_hash=system_output_hash,
                    cost_usd=accumulated_cost,
                    latency_ms=accumulated_latency,
                )
            # Genuine abstain — no prefix, no retry
            return ScoreResult(
                reasoning=reasoning,
                evidence_quotes=evidence_quotes,
                score="Unknown",
                judge_id=judge_id,
                rubric_version=rubric_version,
                prompt_seed=prompt_seed,
                system_output_hash=system_output_hash,
                cost_usd=accumulated_cost,
                latency_ms=accumulated_latency,
            )

        try:
            score_int = int(raw_score)
        except (ValueError, TypeError):
            cause = ABSTAIN_REASON_OUT_OF_RANGE
            if attempt == 0:
                logger.warning(
                    "judge_first_attempt_failure",
                    judge_id=judge_id,
                    item_id=item_id,
                    provider=type(provider).__name__,
                    failure_cause=cause,
                    attempt_index=1,
                )
                continue
            return ScoreResult(
                reasoning=f"{cause}non-int score: {raw_score!r}",
                evidence_quotes=[],
                score="Unknown",
                judge_id=judge_id,
                rubric_version=rubric_version,
                prompt_seed=prompt_seed,
                system_output_hash=system_output_hash,
                cost_usd=accumulated_cost,
                latency_ms=accumulated_latency,
            )

        if score_int not in valid_scores:
            cause = ABSTAIN_REASON_OUT_OF_RANGE
            if attempt == 0:
                logger.warning(
                    "judge_first_attempt_failure",
                    judge_id=judge_id,
                    item_id=item_id,
                    provider=type(provider).__name__,
                    failure_cause=cause,
                    attempt_index=1,
                )
                continue
            return ScoreResult(
                reasoning=(
                    f"{cause}model returned {score_int}, valid levels "
                    f"{sorted(valid_scores)}"
                ),
                evidence_quotes=[],
                score="Unknown",
                judge_id=judge_id,
                rubric_version=rubric_version,
                prompt_seed=prompt_seed,
                system_output_hash=system_output_hash,
                cost_usd=accumulated_cost,
                latency_ms=accumulated_latency,
            )

        # Success
        return ScoreResult(
            reasoning=reasoning,
            evidence_quotes=evidence_quotes,
            score=score_int,
            judge_id=judge_id,
            rubric_version=rubric_version,
            prompt_seed=prompt_seed,
            system_output_hash=system_output_hash,
            cost_usd=accumulated_cost,
            latency_ms=accumulated_latency,
        )

    raise RuntimeError("_call_judge_with_retry: unreachable code path")

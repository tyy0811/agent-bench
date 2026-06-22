"""Repeat harness runs k times per configuration with injected provenance.

This script can make real, PAID API calls. The free path is narrow: a *custom*
entry run with a mock config (provider.default: mock). It is the only path
exercised in CI. langchain entries always build a real ChatOpenAI/ChatAnthropic
from their --provider regardless of any config, so a mock config does NOT make
them free; and a custom entry without a mock config uses its real provider.
run_config_epochs therefore refuses any paid run unless allow_paid is set
(--allow-paid on the CLI), so invoking this script directly cannot silently
spend money -- the Makefile epochs target is not the only guard (guardrail 2).
Provenance is injected by post-processing each entry point's --output JSON into
an envelope file (design spec section 6); harness internals are never edited.
"""

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

# Repo root on path so the preflight can import agent_bench when this file is
# run as a script (python scripts/run_epochs.py puts scripts/ on sys.path[0],
# not the repo root). The heavy work still happens in subprocesses.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

PROVIDER_KEY_ENV = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}

# Provider -> model recorded in langchain config_id (audit #7) AND imported by
# run_langchain_eval to build the client, so the recorded provenance and the
# model that actually bills are the same constant. Do not duplicate it.
MODEL_DEFAULTS = {"openai": "gpt-4o-mini", "anthropic": "claude-haiku-4-5-20251001"}


def _is_mock_config(path: Path | None) -> bool:
    """True only if the config actually sets provider.default: mock.

    The free-path guard must verify config CONTENT, not just that --mock-config
    was passed: a real config handed to --mock-config would otherwise be
    classified free and bill silently (paid-path audit finding #6).
    """
    if path is None:
        return False
    data = yaml.safe_load(Path(path).read_text()) or {}
    return bool(data.get("provider", {}).get("default") == "mock")


def new_ulid() -> str:
    ts = int(datetime.datetime.now(tz=datetime.timezone.utc).timestamp() * 1000)
    time_part = ""
    for _ in range(10):
        time_part = CROCKFORD[ts % 32] + time_part
        ts //= 32
    rand = int.from_bytes(os.urandom(10), "big")
    rand_part = ""
    for _ in range(16):
        rand_part = CROCKFORD[rand % 32] + rand_part
        rand //= 32
    return time_part + rand_part


def write_envelope(
    raw_output: Path,
    dest_dir: Path,
    run_id: str,
    config_id: str,
    epoch: int,
    code_version: str,
    dataset_version: str,
    timestamp: str,
) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    envelope = {
        "run_id": run_id,
        "timestamp": timestamp,
        "config_id": config_id,
        "code_version": code_version,
        "dataset_version": dataset_version,
        "epoch": epoch,
        "results": json.loads(raw_output.read_text()),
    }
    out = dest_dir / f"{config_id.split('+')[0]}_e{epoch}.json"
    out.write_text(json.dumps(envelope, indent=1))
    return out


def _config_hash(path: Path | None) -> str:
    if path is None:
        return "00000000"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:8]


def _dataset_version(golden_path: Path) -> str:
    return "sha-" + hashlib.sha256(golden_path.read_bytes()).hexdigest()[:8]


def _config_id(name: str, spec: dict, mock_config: Path | None) -> str:
    if spec["entry"] == "langchain":
        # Encode the real model so langchain-openai/-anthropic carry meaningful
        # provenance instead of the constant +00000000 sentinel (audit #7); the
        # model is what actually bills. config_id is free-form, schema untouched.
        return f"{name}+{MODEL_DEFAULTS[spec['provider']]}"
    return f"{name}+{_config_hash(mock_config or spec.get('config'))}"


def _code_version() -> str:
    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
    ).stdout.strip()
    return sha + ("-dirty" if dirty else "")


# name -> (entry, config yaml or None, provider flag or None, corpus, golden path)
REGISTRY: dict[str, dict] = {
    "custom-openai": {
        "entry": "custom",
        "config": Path("configs/default.yaml"),
        "corpus": "fastapi",
        "golden": Path("agent_bench/evaluation/datasets/tech_docs_golden.json"),
    },
    "custom-anthropic": {
        "entry": "custom",
        "config": Path("configs/anthropic.yaml"),
        "corpus": "fastapi",
        "golden": Path("agent_bench/evaluation/datasets/tech_docs_golden.json"),
    },
    "langchain-openai": {
        "entry": "langchain",
        "provider": "openai",
        "golden": Path("agent_bench/evaluation/datasets/tech_docs_golden.json"),
    },
    "langchain-anthropic": {
        "entry": "langchain",
        "provider": "anthropic",
        "golden": Path("agent_bench/evaluation/datasets/tech_docs_golden.json"),
    },
    # --- k8s corpus (campaign executed 2026-06-22, paid) --------------------
    # The full 25-question k8s campaign has run: K=5 per config, one run_id each.
    # Envelopes live under results/epochs/ (gitignored, force-added like the
    # fastapi run); tidy rows in results/long/k8s/; the report's k8s section
    # regenerates with `make evaluate-stats`. langchain has no --corpus flag (see
    # _entry_cmd), so k8s is custom-only and the framework-equivalence (TOST)
    # section is empty by construction, not a regression. Re-run with:
    #   make epochs K=5 CONFIGS=custom-openai-k8s,custom-anthropic-k8s CONFIRM_PAID=1
    "custom-openai-k8s": {
        "entry": "custom",
        "config": Path("configs/default.yaml"),
        "corpus": "k8s",
        "golden": Path("agent_bench/evaluation/datasets/k8s_golden.json"),
    },
    "custom-anthropic-k8s": {
        "entry": "custom",
        "config": Path("configs/anthropic.yaml"),
        "corpus": "k8s",
        "golden": Path("agent_bench/evaluation/datasets/k8s_golden.json"),
    },
}


def _entry_cmd(spec: dict, raw_out: Path, mock_config: Path | None) -> list[str]:
    config = mock_config or spec.get("config")
    if spec["entry"] == "custom":
        cmd = [
            sys.executable,
            "scripts/evaluate.py",
            "--mode",
            "deterministic",
            "--output",
            str(raw_out),
        ]
        if config:
            cmd += ["--config", str(config)]
        if spec.get("corpus"):
            cmd += ["--corpus", spec["corpus"]]
        return cmd
    cmd = [
        sys.executable,
        "scripts/run_langchain_eval.py",
        "--provider",
        spec["provider"],
        "--output",
        str(raw_out),
    ]
    if config:
        cmd += ["--config", str(config)]
    return cmd


def run_config_epochs(
    name: str,
    k: int,
    dest_root: Path,
    mock_config: Path | None = None,
    golden_override: Path | None = None,
    allow_paid: bool = False,
) -> list[Path]:
    spec = REGISTRY[name]
    # Guardrail 2 (no silent paid calls): the only free path is a custom entry
    # with a mock config. langchain entries always use a real LLM; a custom
    # entry without a mock config uses its real provider. Refuse otherwise so
    # direct script invocation cannot bill around the Makefile's CONFIRM_PAID.
    if mock_config is not None and not _is_mock_config(mock_config):
        raise SystemExit(
            f"refusing: --mock-config {mock_config} does not set provider.default: mock, "
            "so it would make real (paid) API calls. Pass an actual mock config, "
            "or --allow-paid to confirm you intend to spend money."
        )
    is_free = spec["entry"] == "custom" and mock_config is not None
    if not is_free and not allow_paid:
        raise SystemExit(
            f"refusing: config {name!r} would make real (paid) API calls "
            f"(entry={spec['entry']}, mock_config={'set' if mock_config else 'none'}); "
            "pass --allow-paid to confirm you intend to spend money"
        )
    golden = golden_override or spec["golden"]
    config_id = _config_id(name, spec, mock_config)
    run_id = new_ulid()
    written = []
    for epoch in range(1, k + 1):
        # Raw harness output lives UNDER the run dir, not a sibling results/epochs/raw/,
        # so results/epochs/ contains only run_id dirs. The WP5 convert loop globs
        # results/epochs/*/ and would otherwise feed raw EvalResult lists (no envelope
        # wrapper) to convert_envelopes and crash (audit #4).
        raw_out = dest_root / run_id / "raw" / f"{name}_e{epoch}.json"
        raw_out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(_entry_cmd(spec, raw_out, mock_config), check=True)
        written.append(
            write_envelope(
                raw_output=raw_out,
                dest_dir=dest_root / run_id,
                run_id=run_id,
                config_id=config_id,
                epoch=epoch,
                code_version=_code_version(),
                dataset_version=_dataset_version(golden),
                timestamp=datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
            )
        )
    return written


def _preflight(names: list[str], mock_config: Path | None) -> None:
    """Validate every requested config BEFORE any subprocess runs, so a broken
    config fails at zero cost instead of after earlier configs have already
    billed (paid-path audit finding #2). Checks the registry name, that a custom
    entry's corpus resolves in its config, that the golden file and corpus store
    exist, and that the API key for any paying entry is present in os.environ
    (load_dotenv has already run in main()).
    """
    from agent_bench.core.config import load_config

    problems: list[str] = []
    for name in names:
        spec = REGISTRY.get(name)
        if spec is None:
            problems.append(f"{name}: unknown config (known: {sorted(REGISTRY)})")
            continue
        if not Path(spec["golden"]).exists():
            problems.append(f"{name}: golden dataset missing: {spec['golden']}")
        if spec["entry"] == "custom":
            cfg_path = mock_config or spec.get("config")
            try:
                cfg = load_config(Path(cfg_path) if cfg_path else None)
            except Exception as exc:
                problems.append(f"{name}: config {cfg_path} failed to load ({exc})")
                continue
            provider = cfg.provider.default
            corpus = spec.get("corpus")
            if corpus and corpus not in cfg.corpora:
                problems.append(
                    f"{name}: corpus {corpus!r} not in {cfg_path} corpora "
                    f"{sorted(cfg.corpora)}; evaluate.py would exit 1 mid-run"
                )
            elif corpus and not Path(cfg.corpora[corpus].store_path).exists():
                problems.append(
                    f"{name}: corpus {corpus!r} store {cfg.corpora[corpus].store_path} "
                    "missing; build it (make ingest) before the campaign"
                )
        else:
            provider = spec["provider"]
        free = spec["entry"] == "custom" and _is_mock_config(mock_config)
        if not free and provider != "mock":
            env = PROVIDER_KEY_ENV.get(provider)
            if env and not os.environ.get(env):
                problems.append(f"{name}: {env} not set; a real {provider} run needs it")
    if problems:
        raise SystemExit(
            "preflight failed (no API calls made); fix before spending:\n  - "
            + "\n  - ".join(problems)
        )


def main() -> None:
    from dotenv import load_dotenv

    # Load the gitignored .env so the provider key reads (os.environ, in the
    # subprocesses) and the preflight key check below see the keys (audit #3).
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--configs", required=True, help="comma-separated registry names")
    parser.add_argument("--dest", default="results/epochs")
    parser.add_argument(
        "--mock-config", default=None, help="config YAML forcing provider mock (free)"
    )
    parser.add_argument("--golden", default=None, help="override golden path (tests only)")
    parser.add_argument(
        "--allow-paid",
        action="store_true",
        help="confirm real (paid) API calls for non-mock runs (langchain or no mock config)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run preflight only: validate every config, make no API calls, spend nothing",
    )
    args = parser.parse_args()
    names = args.configs.split(",")
    mock_config = Path(args.mock_config) if args.mock_config else None
    _preflight(names, mock_config=mock_config)
    if args.dry_run:
        print(f"preflight OK for {names}; --dry-run, no API calls made")
        return
    for name in names:
        files = run_config_epochs(
            name,
            args.k,
            Path(args.dest),
            mock_config=mock_config,
            golden_override=Path(args.golden) if args.golden else None,
            allow_paid=args.allow_paid,
        )
        print(f"{name}: wrote {len(files)} epoch envelopes")


if __name__ == "__main__":
    main()

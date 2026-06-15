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

CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


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
    is_free = spec["entry"] == "custom" and mock_config is not None
    if not is_free and not allow_paid:
        raise SystemExit(
            f"refusing: config {name!r} would make real (paid) API calls "
            f"(entry={spec['entry']}, mock_config={'set' if mock_config else 'none'}); "
            "pass --allow-paid to confirm you intend to spend money"
        )
    golden = golden_override or spec["golden"]
    config_id = f"{name}+{_config_hash(mock_config or spec.get('config'))}"
    run_id = new_ulid()
    written = []
    for epoch in range(1, k + 1):
        raw_out = dest_root / "raw" / f"{name}_e{epoch}.json"
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


def main() -> None:
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
    args = parser.parse_args()
    for name in args.configs.split(","):
        files = run_config_epochs(
            name,
            args.k,
            Path(args.dest),
            mock_config=Path(args.mock_config) if args.mock_config else None,
            golden_override=Path(args.golden) if args.golden else None,
            allow_paid=args.allow_paid,
        )
        print(f"{name}: wrote {len(files)} epoch envelopes")


if __name__ == "__main__":
    main()

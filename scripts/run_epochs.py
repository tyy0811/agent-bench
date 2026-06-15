"""Repeat harness runs k times per configuration with injected provenance.

PAID when run against API configs; the Makefile target requires CONFIRM_PAID=1.
The mock config path is free and exercised in CI. Provenance is injected by
post-processing each entry point's --output JSON into an envelope file
(design spec section 6); harness internals are never edited.
"""

import datetime
import json
import os
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

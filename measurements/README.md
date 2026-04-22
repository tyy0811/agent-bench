# measurements/

Raw measurement artifacts referenced from DECISIONS.md entries.

Each file is the raw observation (log snippet, trace, or metric dump)
that backs a specific quantitative claim in DECISIONS.md. Keeping the
raw data here lets a future reader cross-check any DECISIONS.md number
against its underlying evidence without having to re-run the
measurement or trust the narrative summary.

Naming: `YYYY-MM-DD-<topic>-<variant>.log`

Current entries:
- `2026-04-15-coldstart-n1.log`, `-n2.log`, `-n3.log` — HF Spaces cold-start samples N=1..3. Backs the DECISIONS.md entry "Cold-start gate fired — assumption falsified, fix deferred to v1.1 at the right cause."

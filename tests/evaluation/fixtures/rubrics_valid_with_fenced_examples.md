---
dimension: groundedness
scale: binary
reference_based: true
abstain_allowed: true
---

# Groundedness with fenced code examples

## Score 0

Answer adds an unsupported claim.

### Example A — answer references nonexistent score in a code fence

The agent's answer might contain markdown that LOOKS like a section header
but is actually inside a code fence. Example output:

```markdown
## Score 7
This isn't a real rubric level — it's a string that happens to match the
level-header pattern, embedded in a code-fence example.
```

Score=0 because the cited claim above is fabricated; the rubric loader
must not interpret the fenced `## Score 7` as a real level.

## Score 1

Every claim is supported.

### Example B — fenced reference excerpt

The agent might quote a config snippet with a header inside:

```yaml
# Config heading
## Score handler
score_handler: default
```

Score=1 because the fenced YAML is illustrative, not a rubric-structural
header.

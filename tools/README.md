# Optional tools

`research/` contains historical audit and experiment utilities. They are kept
for reproducibility, but are not required by the website or `bpsd-aligner` CLI
and are not included in the installed wheel.

Run one from the repository root with module syntax, for example:

```bash
python -m tools.research.slur_endpoint_check --help
```

New production behavior belongs in `bpsd_aligner/`, not in this directory.

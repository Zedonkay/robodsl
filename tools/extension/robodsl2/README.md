RoboDSL Language Server

This is a Python-based Language Server for RoboDSL using `pygls`.

Features:
- Diagnostics from RoboDSL parser and validator
- Document symbols (nodes, kernels, models)
- Hover (basic word info)
- Completion (keywords and contextual fields)
- Formatting via linter
- Rename (best-effort in-document)
- Semantic tokens (stubbed legend)

Run locally:

```sh
python -m pip install -e .
robodsl-ls
```

Wire into VS Code via a client extension (activate on `robodsl` language) and start the server command `robodsl-ls` over stdio.


# MMLongBench metrics

mmlongbench.py is the unchanged official evaluation source from MMLongBench-Doc:
https://github.com/Mayuyu-Mio/MMLongBench-Doc

The copy is taken from this project's previously frozen eval_score.py. Its SHA-256 is recorded in docs/layout-and-agent.md.
It is a benchmark scoring utility, not an Agent implementation.
The scoring wrapper parses list literals with ast.literal_eval before calling it; GT is used only offline.
The official function's empty-list and malformed-input limitations are reported as scoring errors rather than silently changing the scoring rule.

# Source-free memory: results

## Completed pilot — reviewed 2026-11-28

Artifacts: `artifacts/pilot-none-shuffled/`. Validation NLL selected
`last_token`, layer 28, epoch 5, with frozen Qwen3-4B and a four-slot adapter.

| Metric | Memory | Question only |
|---|---:|---:|
| Mean balanced accuracy | 55.06% | 47.62% |
| All seven answers correct | 4/96 | 0/96 |
| Invalid answers | 0/672 | 0/672 |

**The source-free interface gives a limited positive signal (+7.44 percentage
points), but reliable use of a coherent spatial state was not demonstrated.**
Memory answers contain at least one contradictory inverse-relation pair in
82/96 situations. Formatting failures do not explain the weak performance.

For Direction C, a saved situation vector delivered through a learned adapter
can improve answers without the source text. This does not establish a prepared
or causally used world state: the adapter has 26.2 million trainable parameters,
memory-swap/null/oracle controls are absent, and only four independent test
layouts are available. Representation, interface, and training limitations
remain possible explanations for the weak result.

Both sets of 672 answers were checked against dataset labels; recomputed metrics
match the report. Dataset and checkpoint hashes were verified. Feature extraction
and model inference were not rerun; the source feature cache was unavailable.

# Paper Status

## Submission checklist

| Field | Status |
| --- | --- |
| Title | WHEN DOES TEMPORAL FEATURE REUSE ACCELERATE VIDEO SEMANTIC SEGMENTATION? A HARDWARE-AWARE STUDY OF REUSE BOUNDARIES |
| Target | IEEE ICIP 2027 |
| Format baseline | ICIP 2026 official kit; final 2027 kit recheck pending |
| Source status | PASS |
| Abstract word count | 128 whitespace-separated words |
| Technical page count | 4 |
| Reference-only page count | 1 (page 5) |
| Total pages | 5 |
| Figure count | 4 |
| Table count | 3 manuscript tables; generated CSV tables retained as supplements |
| Citation count | 11 unique keys |
| Verified references | 11 / 11 |
| Undefined citations | 0 in final builds |
| Undefined references | 0 in final builds |
| Overfull boxes | 0 |
| Underfull boxes | blind: 1 hbox + 1 vbox; publish: 1 hbox + 2 vbox; non-fatal official-style warnings |
| Blind audit | PASS: no identity hits in extracted PDF text; no Author metadata |
| Author information | TODO; publish PDF contains placeholders only |
| Final ICIP 2027 author kit recheck | TODO |
| Cache memory audit | PASS |
| Pareto audit | PASS |
| Claim audit | PASS |
| Citation audit | PASS |
| Reviewer #2 simulation | PASS; no fatal scientific flaw; novelty risk remains borderline |
| Reviewed PDFs | `output/icip2027_blind_reviewed.pdf`, `output/icip2027_publish_reviewed.pdf` |

## Build status

Both final PDFs were compiled with MiKTeX 25.12 pdfTeX using the standard
`pdflatex`, `bibtex`, `pdflatex`, `pdflatex` sequence. Both are US Letter with
embedded fonts. Build commands and logs are documented in `BUILD.md` and
`output/`.

## Scientific boundary

The manuscript reports only the frozen Round 10 VSPW validation results for
SegFormer-B0/B1 on one RTX 4090 D protocol. Round 5 numerical microbenchmark
values are excluded from quantitative tables and retained only as separately
scoped qualitative context. Cityscapes, edge hardware, new models, and new
experiments are outside the paper.

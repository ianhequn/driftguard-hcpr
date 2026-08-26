# PDF Audit

Date: 2026-08-26

| File | Pages | Page size | Fonts embedded | Identity scan |
| --- | ---: | --- | --- | --- |
| `output/icip2027_blind_round10.pdf` | 5 | 612 x 792 pt, US Letter | yes | PASS |
| `output/icip2027_publish_round10.pdf` | 5 | 612 x 792 pt, US Letter | yes | placeholder only |

Both files were built with MiKTeX 25.12 pdfTeX and the standard four-step
sequence (`pdflatex`, `bibtex`, `pdflatex`, `pdflatex`). The technical content
occupies pages 1--4; the references begin on page 5.

## Warning checks

```text
Undefined citations: 0
Undefined references: 0
Overfull hbox: 0
Blind: 1 underfull hbox, 1 underfull vbox
Publish: 1 underfull hbox, 2 underfull vbox
```

The remaining underfull warnings are non-fatal layout messages from the
official two-column style. The rendered pages contain no clipping or identity
leakage.

## Blind metadata and text

`pdfinfo` reports no `Author` field for the blind PDF. Extracted blind text has
no author name, institution, email, GitHub, university, funding, or
acknowledgment identifier. The publish PDF intentionally contains only the
placeholder identity defined in `main.tex`.

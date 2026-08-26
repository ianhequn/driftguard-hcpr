# PDF Audit

Date: 2026-08-26

## File checks

| File | Pages | Page size | Fonts embedded | Identity scan |
| --- | ---: | --- | --- | --- |
| `output/icip2027_blind_draft.pdf` | 3 | 612 x 792 pt, US Letter | yes | PASS |
| `output/icip2027_publish_draft.pdf` | 3 | 612 x 792 pt, US Letter | yes | placeholder only |

Both files were built with MiKTeX 25.12 pdfTeX 3.141592653-2.6-1.40.28.
The standard sequence was `pdflatex`, `bibtex`, `pdflatex`, `pdflatex`.

## Warning checks

```text
Undefined citations: 0
Undefined references: 0
Overfull hbox: 0
Blind underfull hbox: 2
Publish underfull hbox: 2
Publish underfull vbox: 2
```

The underfull messages are the non-fatal index/contribution paragraph and
column-balancing warnings emitted by the official two-column style. No text,
figure, or table is clipped in the visual rendering.

## Page policy

The manuscript is 3 pages total. Technical content and references share the
last page; page 6 is not present, so the references-only sixth-page rule is
not used.

## Blind metadata and text

`pdfinfo` reports no `Author` field for the blind PDF. Extracted blind text has
no hit for author name, institution, email, GitHub, university, funding, or
acknowledgment identifiers. The publish PDF intentionally contains only the
placeholder identity defined in `main.tex`.

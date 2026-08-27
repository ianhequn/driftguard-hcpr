# MMM 2027 HCPR Revised Anonymous Manuscript

This directory contains the revised anonymous manuscript reconstructed from
`MMM2027_HCPR_anonymous_audited.pdf` and updated against the requested
reviewer-risk list.

## Build

Run the following in this directory:

```text
pdflatex -interaction=nonstopmode -halt-on-error MMM2027_HCPR_anonymous_revised.tex
bibtex MMM2027_HCPR_anonymous_revised
pdflatex -interaction=nonstopmode -halt-on-error MMM2027_HCPR_anonymous_revised.tex
pdflatex -interaction=nonstopmode -halt-on-error MMM2027_HCPR_anonymous_revised.tex
```

The package includes the local `llncs.cls`, `splncs04.bst`, bibliography, and
the two figures used by the manuscript. No dataset, checkpoint, raw prediction,
or profiler dump is included.

## Revision boundary

The revision adds published quantitative TFC positioning, absolute Full mIoU,
the complete five-point fixed-policy reference set, the mVC definition, the
OldCost/NewCost sensitivity interpretation, uppercase `Q` notation, and the
Q80-to-Q90 confidence-interval explanation. The TFC values are explicitly
marked as external because the archived HCPR results do not contain the
per-frame masks needed for a matched mVC8/mVC16 recomputation.

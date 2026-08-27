# ICASSP 2027 HCPR Layout Audit

## Build

- Source: `paper/icip2027/Hequn_Lan_ICASSP2027_HCPR.tex`
- Template: local `spconf.sty`, official SPS two-column layout
- Engine: MiKTeX 25.12 pdfTeX
- Bibliography: local `IEEEbib.bst` and verified `references.bib`
- Final PDF: 4 pages, US Letter
- Abstract: 141 words
- Technical content and references remain within the four-page submission file.

## Automated checks

- Undefined citations: 0
- Undefined references: 0
- Fatal build errors: 0
- Placeholder author/email strings: 0
- Non-ASCII source characters: 0
- Overfull boxes: 0
- Remaining messages: two ordinary underfull hbox messages in the abstract/
  contribution text and one underfull vbox from the official float layout;
  visual inspection found no clipped or overlapping content.

## Visual checks

- Title, author, affiliation, abstract, and index terms render on page 1.
- HCPR method diagram is legible and contained within the column.
- Cross-device table and Pareto figure are aligned and readable.
- Physical-call and stage-cost tables do not overlap the body text.
- The conclusion is kept as one visual block; references continue cleanly.
- No black overfull markers, clipped text, broken glyphs, or missing figures.

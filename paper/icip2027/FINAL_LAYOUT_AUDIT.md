# Final Layout Audit

## Previous visual issues

- Page 4 had excessive unused space around Fig. 4, the late analysis, and the conclusion.
- Fig. 2 was too small relative to its role as the central result.
- Fig. 1 used more space than needed for a simple execution schematic.
- Fig. 3 overlapped conceptually with Table 2 and Fig. 2.
- Table 2 used separate CI low/high columns and consumed unnecessary width.
- The reference list appeared visually one-sided and required a template-level audit.

## Changes made

- Fig. 1 was regenerated at a shorter height while preserving the fresh-prefix/cached-suffix labels.
- Fig. 2 was changed to a wider two-column float and remains adjacent to the main result discussion.
- Fig. 3 quality-versus-refresh was removed from the manuscript; its scientific observation remains in the results prose.
- Fig. 4 was slightly reduced to remain a compact single-column hardware interpretation figure.
- Table 1 now uses the layout-safe heading “Refresh-only suffix calls”; values and semantics are unchanged.
- Table 2 keeps all 18 rows and combines CI low/high into one 95% CI column without changing any numeric value.
- Table 3 remains compact and continues to separate runtime share, logical cache, and allocated cache.
- The standard template two-column flow was retained. No margins, body font size, or scientific content were changed.

## Figure decisions

- Fig. 1: KEEP / RESIZED
- Fig. 2: KEEP / ENLARGED
- Fig. 3: REMOVED
- Fig. 4: KEEP / RESIZED

## Table changes

Table 1 retains the full physical-call evidence and clarifies that suffix calls occur only on refresh frames. Table 2 retains every finalized setting and uses a single interval column. Table 3 is unchanged in values and continues to label the logical cache columns separately from the allocated persistent-cache sentence.

## Page-by-page audit

### Page 1

Title, abstract, introduction, related work, and the beginning of the execution contract are clean. The title wraps across three centered lines without clipping. Equations and section hierarchy are readable.

### Page 2

The compact Fig. 1, physical-call Table 1, experimental protocol, and the beginning of results form a coherent method-to-measurement flow. The table is readable at normal zoom and its heading is unambiguous.

### Page 3

The complete 18-row Table 2 and enlarged Fig. 2 are the dominant visual elements. Table 3 follows as supporting runtime/cache evidence. Fig. 2 labels and grayscale marker shapes remain readable.

### Page 4

Fig. 3 in the final layout numbering is the compact runtime-opportunity plot, followed by correctness, cache accounting, limitations, and conclusion. The column flow is valid and unclipped. Some lower-page whitespace remains because the bibliography is intentionally separated onto page 5; it is acceptable under the no-filler constraint.

### Page 5

References begin at the top in the normal template two-column environment. The eleven references fit within the left column, so the right column remains unused; this is natural flow, not an accidental one-column command.

## Reference-column fix

spconf.sty activates the standard two-column environment, and the manuscript contains no one-column command or manual column break. The bibliography therefore uses normal template flow. A separate page is retained with clearpage so the final structure is four technical pages plus one reference page.

## Scientific diff audit

The layout pass changed only figure inclusion/sizing, table formatting labels, CI presentation, and float/page flow.

- All 18 Table 2 numeric rows: unchanged.
- Full B0/B1 mIoU: unchanged.
- S4/FI-2 and S2--S4/FI-8 headline values: unchanged.
- B0/B1 five-point Pareto frontier: unchanged.
- Runtime shares, cache values, dataset statistics, and timing protocol: unchanged.
- Scientific conclusions and claim boundary: unchanged.

SCIENTIFIC CONTENT CHANGED = NO

## Remaining visual limitations

The official style emits one non-fatal underfull hbox warning in each final build and no overfull boxes. The reference page has unused right-column space because the verified bibliography is short. Author placeholders remain in the publish build until authorship is supplied. Final ICIP 2027 kit checking remains a human action.

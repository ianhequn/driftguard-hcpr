# HCPR ICASSP 2027 Source Package

This directory contains the independent HCPR manuscript source. The older
ICIP/round-10 sources (`main.tex`, `blind.tex`, and `publish.tex`) are kept
unchanged.

## Build

Run from this directory with MiKTeX or another LaTeX installation:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error Hequn_Lan_ICASSP2027_HCPR.tex
bibtex Hequn_Lan_ICASSP2027_HCPR
pdflatex -interaction=nonstopmode -halt-on-error Hequn_Lan_ICASSP2027_HCPR.tex
pdflatex -interaction=nonstopmode -halt-on-error Hequn_Lan_ICASSP2027_HCPR.tex
```

The source uses the local SPS template `spconf.sty`, bibliography style
`IEEEbib.bst`, and `references.bib`. The two included figures are generated
from frozen Round 11A/11B CSV artifacts by `generate_hcpr_assets.py`; figure
generation does not run inference or change any experiment.

## Scope

The manuscript reports the frozen HCPR results on VSPW validation with
SegFormer-B0/B1 and RTX 4090 D/RTX 3080 Ti. It does not include datasets,
checkpoints, raw videos, profiler traces, or result dumps.

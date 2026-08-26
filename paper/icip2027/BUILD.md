# Build Instructions

The manuscript uses the official `spconf.sty` and `IEEEbib.bst` files stored in this directory.

## Blind PDF

```powershell
cd D:\DriftGuard\paper\icip2027
pdflatex -interaction=nonstopmode -halt-on-error -jobname=icip2027_blind_draft blind.tex
bibtex icip2027_blind_draft
pdflatex -interaction=nonstopmode -halt-on-error -jobname=icip2027_blind_draft blind.tex
pdflatex -interaction=nonstopmode -halt-on-error -jobname=icip2027_blind_draft blind.tex
Move-Item -Force icip2027_blind_draft.pdf output\icip2027_blind_draft.pdf
```

## Publish PDF

```powershell
pdflatex -interaction=nonstopmode -halt-on-error -jobname=icip2027_publish_draft publish.tex
bibtex icip2027_publish_draft
pdflatex -interaction=nonstopmode -halt-on-error -jobname=icip2027_publish_draft publish.tex
pdflatex -interaction=nonstopmode -halt-on-error -jobname=icip2027_publish_draft publish.tex
Move-Item -Force icip2027_publish_draft.pdf output\icip2027_publish_draft.pdf
```

The exact shell quoting may need adjustment on a remote Linux host. The important contract is that the blind build defines `blindversion` and the publish build defines `publishversion`.

## Verification

After compilation, inspect page count, fonts, metadata, and anonymization. The current Windows environment did not expose `pdflatex`, `latexmk`, or `tectonic` during the initial audit, so compilation remains an explicit gate.

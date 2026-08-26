# Build Instructions

The manuscript uses the official `spconf.sty` and `IEEEbib.bst` files stored in this directory.

## Build environment and result

```text
OS: Windows PowerShell
Date: 2026-08-26
Compiler: MiKTeX 25.12 / pdfTeX 3.141592653-2.6-1.40.28
Template: official spconf.sty and IEEEbib.bst
Result: blind and publish builds PASS
Pages: 3 each
```

`latexmk` is installed but cannot run because its Perl script engine is not
available. The standard three-pass sequence below was used instead.

## Blind PDF

```powershell
cd D:\DriftGuard\paper\icip2027
pdflatex -interaction=nonstopmode -halt-on-error -jobname=icip2027_blind_draft blind.tex
bibtex icip2027_blind_draft
pdflatex -interaction=nonstopmode -halt-on-error -jobname=icip2027_blind_draft blind.tex
pdflatex -interaction=nonstopmode -halt-on-error -jobname=icip2027_blind_draft blind.tex
Move-Item -Force icip2027_blind_draft.pdf output\icip2027_blind_draft.pdf
Copy-Item -Force icip2027_blind_draft.log output\blind_build.log
```

## Publish PDF

```powershell
pdflatex -interaction=nonstopmode -halt-on-error -jobname=icip2027_publish_draft publish.tex
bibtex icip2027_publish_draft
pdflatex -interaction=nonstopmode -halt-on-error -jobname=icip2027_publish_draft publish.tex
pdflatex -interaction=nonstopmode -halt-on-error -jobname=icip2027_publish_draft publish.tex
Move-Item -Force icip2027_publish_draft.pdf output\icip2027_publish_draft.pdf
Copy-Item -Force icip2027_publish_draft.log output\publish_build.log
```

The exact shell quoting may need adjustment on a remote Linux host. The important contract is that the blind build defines `blindversion` and the publish build defines `publishversion`.

## Verification

## Audit commands and results

```powershell
pdfinfo output\icip2027_blind_draft.pdf
pdfinfo output\icip2027_publish_draft.pdf
pdffonts output\icip2027_blind_draft.pdf
pdffonts output\icip2027_publish_draft.pdf
pdftotext output\icip2027_blind_draft.pdf -
```

Both PDFs are US Letter, 3 pages, and all listed fonts are embedded. The blind
PDF has no author, institution, email, GitHub, funding, or university hit in
extracted text; its metadata has no `Author` field. The publish PDF contains
only the explicit placeholder identity from `main.tex`. Final build logs have
zero undefined citations, zero undefined references, and zero overfull boxes.
The remaining underfull hbox/vbox messages are non-fatal layout warnings and
are recorded in `PAPER_STATUS.md`.

There is no page 6; the page-6 references-only rule is therefore not used.

# Build Instructions

The manuscript uses the official `spconf.sty` and `IEEEbib.bst` files stored in
this directory.

## Final build

MiKTeX 25.12 / pdfTeX was used on Windows PowerShell. `pdflatex.exe` and
`bibtex.exe` are under the local MiKTeX `bin\\x64` directory. The blind and
publish sources define `blindversion` and `publishversion`, respectively.

```powershell
$env:MIKTEX_ENABLEINSTALLER='0'
pdflatex -interaction=nonstopmode -halt-on-error -jobname=icip2027_blind_round10 blind.tex
bibtex icip2027_blind_round10
pdflatex -interaction=nonstopmode -halt-on-error -jobname=icip2027_blind_round10 blind.tex
pdflatex -interaction=nonstopmode -halt-on-error -jobname=icip2027_blind_round10 blind.tex

pdflatex -interaction=nonstopmode -halt-on-error -jobname=icip2027_publish_round10 publish.tex
bibtex icip2027_publish_round10
pdflatex -interaction=nonstopmode -halt-on-error -jobname=icip2027_publish_round10 publish.tex
pdflatex -interaction=nonstopmode -halt-on-error -jobname=icip2027_publish_round10 publish.tex
```

Final outputs:

```text
output/icip2027_blind_round10.pdf
output/icip2027_publish_round10.pdf
```

## Verification

The final files are US Letter, five pages, and use embedded fonts. The blind
PDF contains no author name, institution, email, GitHub, university, funding,
or acknowledgment identity. References begin on page 5. Final logs contain no
undefined citations, undefined references, or overfull boxes. The remaining
underfull hbox/vbox messages are non-fatal official-style layout warnings.

# DriftGuard Round 9 Sequence HTTP Audit

## Result

```text
HTTP RANGE: UNKNOWN
ZIP FORMAT: UNKNOWN
REMOTE CENTRAL DIRECTORY: NOT RETRIEVED
```

The logged-in browser can display the official package list, but opening the
package handler was blocked by the browser client. A command-line Range probe
to the official package handler, without browser authentication, returned an
HTTP redirect to the official login page. It did not reach the archive object,
so this is not evidence that the archive itself supports or rejects Range.

No archive body, tail bytes, central directory, or entry payload was
downloaded. The probe details are stored in
`results/R9R_E03_sequence_http_audit/http_metadata.json`.

```text
REMOTE ZIP SELECTIVE EXTRACTION = NOT YET EVALUATED
FULL 324 GB DOWNLOAD = FORBIDDEN
```

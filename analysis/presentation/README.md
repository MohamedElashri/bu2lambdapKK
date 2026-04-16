# Presentation Workflows

This directory holds optional note, slide, and validation plotting workflows.

These workflows are intentionally separate from the default physics run:

- they are not part of the core reproducible yield/efficiency/systematics chain
- they consume stable pipeline products where possible
- when they need raw tuples, they should read shared assumptions from
  `modules/presentation_config.py`

Current subareas:

- `ana_note_plots/`
  - note-facing figure production
- `background_studies/`
  - background-study figure production
- `validation/`
  - presentation-only plotting tied to active study products

Run via the top-level entrypoints from `analysis/`:

```bash
make plots
make plot-note
make plot-backgrounds
make plot-reweighting
```

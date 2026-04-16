# Tracking Systematic

This is a documentation-only standalone study retained for provenance. It
records the rationale for treating the tracking systematic as a ratio
cancellation assumption rather than as a generated workflow product.

## Status

- standalone documentation only
- not part of the top-level Snakemake DAG
- referenced conceptually by the systematic aggregation, but not executed

## Retained Source

- `tracking_systematic.md`

## Relationship To The Main Workflow

The active systematic aggregator still treats tracking as a documented
assumption rather than a computed artifact. This directory stays under
`analysis/studies/standalone/` so that rationale remains discoverable without
pretending it is an active pipeline stage.

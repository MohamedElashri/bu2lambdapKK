"""
Sideband Background Modeling Module

This module provides tools for data-driven combinatorial background modeling
using B+ mass sideband regions in the Bu -> Lambda_bar p K+ K- analysis.

Modules:
--------
- validate_shapes: Validate M(LpK-) shape independence across M(B+) sidebands
- extract_template: Extract background template from sideband data
- template_fitter: Perform fits using sideband-derived background templates
"""

from pathlib import Path

# Module directory
MODULE_DIR: Path = Path(__file__).parent
ANALYSIS_DIR: Path = MODULE_DIR.parent.parent

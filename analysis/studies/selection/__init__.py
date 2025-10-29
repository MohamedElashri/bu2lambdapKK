"""
Selection Study Package for B+ → pK⁻Λ̄ K+ Analysis

This package provides modular components for selection optimization studies:
- efficiency: Efficiency calculations and cutflow analysis
- variable_analyzer: Individual variable distributions and efficiency scans
- plot: Plotting utilities and wrappers
- jpsi_analyzer: J/ψ mass spectrum analysis
- main: Main orchestrator and entry point

Author: Mohamed Elashri
Date: October 28, 2025
"""

__version__ = '1.0.0'
__author__ = 'Mohamed Elashri'

from .selection_efficiency import EfficiencyCalculator
from .variable_analyzer import VariableAnalyzer
from .plot import StudyPlotter
from .jpsi_analyzer import JPsiAnalyzer

__all__ = [
    'EfficiencyCalculator',
    'VariableAnalyzer',
    'StudyPlotter',
    'JPsiAnalyzer'
]

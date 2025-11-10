#!/usr/bin/env python3
"""
Custom exceptions for the Bu2lambdapKK analysis pipeline

Provides a hierarchy of exceptions for better error handling and diagnostics.
All custom exceptions inherit from AnalysisError for easy catching.
"""


class AnalysisError(Exception):
    """
    Base exception for all analysis pipeline errors
    
    All custom exceptions inherit from this class, allowing users to catch
    all analysis-specific errors with a single except clause.
    """
    pass


class ConfigurationError(AnalysisError):
    """
    Raised when configuration is invalid or missing required fields
    
    Examples:
    - Missing required config file
    - Invalid parameter values
    - Missing required config sections
    """
    pass


class DataLoadError(AnalysisError):
    """
    Raised when data or MC files cannot be loaded
    
    Examples:
    - File not found
    - Corrupted ROOT file
    - Missing tree in ROOT file
    - Empty dataset
    """
    pass


class BranchMissingError(AnalysisError):
    """
    Raised when required branch is not found in data
    
    Examples:
    - Missing physics branch (e.g., Bu_PT, L0_MM)
    - Branch name typo in configuration
    - Data format mismatch
    """
    def __init__(self, branch_name: str, file_path: str = None):
        """
        Initialize BranchMissingError
        
        Args:
            branch_name: Name of the missing branch
            file_path: Optional path to the file being read
        """
        self.branch_name = branch_name
        self.file_path = file_path
        
        message = f"Required branch '{branch_name}' not found"
        if file_path:
            message += f" in file: {file_path}"
        
        super().__init__(message)


class OptimizationError(AnalysisError):
    """
    Raised when cut optimization fails
    
    Examples:
    - No valid cut combinations found
    - FOM calculation fails
    - Insufficient statistics
    """
    pass


class FittingError(AnalysisError):
    """
    Raised when mass fitting fails
    
    Examples:
    - Fit does not converge
    - Invalid fit parameters
    - Insufficient data for fit
    """
    pass


class EfficiencyError(AnalysisError):
    """
    Raised when efficiency calculation fails
    
    Examples:
    - Division by zero (no generated events)
    - Negative efficiency (bug in calculation)
    - Missing efficiency data for state/year
    """
    pass


class ValidationError(AnalysisError):
    """
    Raised when validation checks fail
    
    Examples:
    - Inconsistent yields across years
    - Unphysical results (e.g., negative yields)
    - Failed sanity checks
    """
    pass


class CacheError(AnalysisError):
    """
    Raised when cache operations fail
    
    Examples:
    - Cannot write cache file
    - Corrupted cache
    - Cache version mismatch
    """
    pass

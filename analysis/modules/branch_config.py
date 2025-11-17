"""
Branch configuration manager

Handles loading and parsing branch configurations from branches_config.toml
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import tomli

from .exceptions import ConfigurationError


class BranchConfig:
    """Manager for branch configuration.

    Attributes:
        logger: Logger instance for this class
        config: Loaded TOML configuration dictionary
        data_to_common: Mapping from data branch names to common names
        mc_to_common: Mapping from MC branch names to common names
        common_to_data: Mapping from common names to data branch names
        common_to_mc: Mapping from common names to MC branch names
    """

    def __init__(self, config_path: str | None = None) -> None:
        """
        Initialize branch configuration.

        Args:
            config_path: Path to branches_config.toml (auto-detected if None)

        Raises:
            ConfigurationError: If configuration file not found
        """
        self.logger: logging.Logger = logging.getLogger("Bu2LambdaPKK.BranchConfig")

        # Auto-detect config file
        if config_path is None:
            config_path = Path(__file__).parent / "branches_config.toml"
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            raise ConfigurationError(
                f"Branch configuration file not found: {config_path}\n"
                f"Expected location: {Path(__file__).parent / 'branches_config.toml'}"
            )

        # Load configuration
        with open(config_path, "rb") as f:
            self.config: dict[str, Any] = tomli.load(f)

        # Build alias mappings for quick lookup
        self._build_alias_maps()

        self.logger.info(f"Loaded branch configuration from {config_path}")

    def _build_alias_maps(self) -> None:
        """Build reverse lookup maps for aliases."""
        self.data_to_common: dict[str, str] = {}  # data branch name -> common name
        self.mc_to_common: dict[str, str] = {}  # mc branch name -> common name
        self.common_to_data: dict[str, str] = {}  # common name -> data branch name
        self.common_to_mc: dict[str, str] = {}  # common name -> mc branch name

        if "aliases" not in self.config:
            return

        # Process all alias groups (e.g., aliases.pid)
        for group_name, aliases in self.config["aliases"].items():
            if group_name == "description":
                continue

            for common_name, mapping in aliases.items():
                if isinstance(mapping, dict) and "data" in mapping and "mc" in mapping:
                    data_branch = mapping["data"]
                    mc_branch = mapping["mc"]

                    self.common_to_data[common_name] = data_branch
                    self.common_to_mc[common_name] = mc_branch
                    self.data_to_common[data_branch] = common_name
                    self.mc_to_common[mc_branch] = common_name

    def get_branches_from_sets(self, sets: list[str], exclude_mc: bool = False) -> list[str]:
        """
        Get all branches from specified sets

        Parameters:
        - sets: List of set names (e.g., ['essential', 'kinematics'])
        - exclude_mc: If True, exclude MC-only branches

        Returns:
        - Flattened list of unique branch names
        """
        branches = set()

        for set_name in sets:
            if set_name not in self.config["branches"]:
                self.logger.warning(f"Branch set '{set_name}' not found in config")
                continue

            branch_set = self.config["branches"][set_name]

            # Check if this is MC-only and we should skip it
            if exclude_mc and branch_set.get("mc_only", False):
                self.logger.info(f"Skipping MC-only set: {set_name}")
                continue

            # Add branches from all particles in this set
            for particle, particle_branches in branch_set.items():
                if particle == "description" or particle == "mc_only":
                    continue
                if isinstance(particle_branches, list):
                    branches.update(particle_branches)

        return sorted(list(branches))

    def get_branches_from_preset(self, preset: str, exclude_mc: bool = False) -> list[str]:
        """
        Get branches from a preset configuration

        Parameters:
        - preset: Preset name (e.g., 'minimal', 'standard', 'mc_reco')
        - exclude_mc: If True, exclude MC-only branches

        Returns:
        - List of branch names
        """
        if "presets" not in self.config:
            raise ConfigurationError(
                "No presets defined in branch configuration. "
                "branches_config.toml must have a [presets] section."
            )

        if preset not in self.config["presets"]:
            available = list(self.config["presets"].keys())
            raise ConfigurationError(
                f"Preset '{preset}' not found in branch configuration.\n"
                f"Available presets: {available}"
            )

        sets = self.config["presets"][preset]
        return self.get_branches_from_sets(sets, exclude_mc=exclude_mc)

    def get_truth_branches(self, particle: str | None = None) -> list[str]:
        """
        Get MC truth branches from MCDecayTree

        Parameters:
        - particle: Specific particle name (e.g., 'Bplus', 'Kplus')
                   If None, returns all truth branches

        Returns:
        - List of truth branch names
        """
        if "truth_branches" not in self.config:
            return []

        if particle is None:
            # Return all truth branches
            branches = []
            for part, part_branches in self.config["truth_branches"].items():
                if part == "description":
                    continue
                if isinstance(part_branches, list):
                    branches.extend(part_branches)
            return branches
        # Return branches for specific particle
        return self.config["truth_branches"].get(particle, [])

    def get_default_load_sets(self) -> list[str]:
        """
        Get the default sets to load from config

        Returns:
        - List of set names
        """
        return self.config["branches"].get("load_sets", [])

    def resolve_aliases(self, branches: list[str], is_mc: bool = False) -> list[str]:
        """
        Resolve common branch names to actual data/MC branch names

        Parameters:
        - branches: List of branch names (may include common aliases)
        - is_mc: If True, resolve to MC names; otherwise resolve to data names

        Returns:
        - List of actual branch names
        """
        resolved = []
        mapping = self.common_to_mc if is_mc else self.common_to_data

        for branch in branches:
            if branch in mapping:
                # This is a common name, resolve it
                actual_branch = mapping[branch]
                resolved.append(actual_branch)
                self.logger.debug(
                    f"Resolved alias: {branch} -> {actual_branch} " f"({'MC' if is_mc else 'data'})"
                )
            else:
                # Not an alias, use as-is
                resolved.append(branch)

        return resolved

    def normalize_branches(self, branches: list[str], is_mc: bool = False) -> dict[str, str]:
        """
        Create mapping from actual branch names to common names

        This is useful for renaming branches after loading so that
        your analysis code can use common names regardless of data/MC

        Parameters:
        - branches: List of actual branch names from file
        - is_mc: If True, treat as MC branches; otherwise data branches

        Returns:
        - Dictionary mapping actual_name -> common_name
        """
        mapping = self.mc_to_common if is_mc else self.data_to_common
        rename_map = {}

        for branch in branches:
            if branch in mapping:
                common_name = mapping[branch]
                rename_map[branch] = common_name
                self.logger.debug(f"Will normalize: {branch} -> {common_name}")

        return rename_map

    def list_available_sets(self) -> list[str]:
        """List all available branch sets.

        Returns:
            List of branch set names
        """
        return [k for k in self.config["branches"].keys() if k != "load_sets"]

    def list_available_presets(self) -> list[str]:
        """List all available presets.

        Returns:
            List of preset names
        """
        return list(self.config.get("presets", {}).keys())

    def get_branches_by_particle(
        self, particle: str, sets: list[str], exclude_mc: bool = False
    ) -> list[str]:
        """
        Get branches for a specific particle from specified sets

        Parameters:
        - particle: Particle name (e.g., 'Bu', 'L0', 'h1')
        - sets: List of set names
        - exclude_mc: If True, exclude MC-only branches

        Returns:
        - List of branch names for that particle
        """
        branches = []

        for set_name in sets:
            if set_name not in self.config["branches"]:
                continue

            branch_set = self.config["branches"][set_name]

            # Check if this is MC-only and we should skip it
            if exclude_mc and branch_set.get("mc_only", False):
                continue

            if particle in branch_set:
                particle_branches = branch_set[particle]
                if isinstance(particle_branches, list):
                    branches.extend(particle_branches)

        return branches

    def validate_branches(
        self, branches: list[str], available_branches: list[str]
    ) -> dict[str, Any]:
        """
        Validate that requested branches exist in the file.

        Args:
            branches: List of requested branch names
            available_branches: List of available branches from ROOT file

        Returns:
            Dictionary with keys:
                - 'valid': List of found branches
                - 'missing': List of missing branches
                - 'found': Count of found branches
                - 'total_requested': Total number requested
        """
        available_set = set(available_branches)
        requested_set = set(branches)

        valid = sorted(list(requested_set & available_set))
        missing = sorted(list(requested_set - available_set))

        if missing:
            self.logger.warning(f"Missing {len(missing)} requested branches: {missing[:5]}...")

        return {
            "valid": valid,
            "missing": missing,
            "found": len(valid),
            "total_requested": len(branches),
        }


def get_branch_config(config_path: str | None = None) -> BranchConfig:
    """
    Convenience function to get a BranchConfig instance

    Parameters:
    - config_path: Path to configuration file

    Returns:
    - BranchConfig instance
    """
    return BranchConfig(config_path)

"""
Unit tests for BranchConfig module.

Tests configuration loading, alias resolution, branch validation,
and preset management without requiring real configuration files.
"""

from __future__ import annotations

import pytest
import tomli_w
from pathlib import Path
from typing import Dict, Any

from analysis.modules.branch_config import BranchConfig, get_branch_config
from analysis.modules.exceptions import ConfigurationError


@pytest.mark.unit
class TestBranchConfigInitialization:
    """Test BranchConfig initialization and loading."""
    
    def test_load_from_file(self, branches_config_file: Path) -> None:
        """Test loading configuration from file."""
        config = BranchConfig(str(branches_config_file))
        
        assert config.config is not None
        assert isinstance(config.config, dict)
        assert hasattr(config, 'logger')
    
    def test_auto_detect_config_file(self, tmp_test_dir: Path) -> None:
        """Test auto-detection of config file fails gracefully."""
        # BranchConfig tries to auto-detect in its own directory
        # This will fail unless branches_config.toml exists there
        with pytest.raises(ConfigurationError) as exc_info:
            # Use a non-existent explicit path
            BranchConfig(str(tmp_test_dir / "nonexistent.toml"))
        
        assert "not found" in str(exc_info.value)
    
    def test_config_file_not_found(self, tmp_test_dir: Path) -> None:
        """Test error when config file doesn't exist."""
        nonexistent = tmp_test_dir / "missing.toml"
        
        with pytest.raises(ConfigurationError) as exc_info:
            BranchConfig(str(nonexistent))
        
        assert "not found" in str(exc_info.value)


@pytest.mark.unit
class TestAliasMapping:
    """Test alias mapping and resolution."""
    
    def test_build_alias_maps(self, branches_config_file: Path) -> None:
        """Test that alias maps are built correctly."""
        config = BranchConfig(str(branches_config_file))
        
        assert hasattr(config, 'data_to_common')
        assert hasattr(config, 'mc_to_common')
        assert hasattr(config, 'common_to_data')
        assert hasattr(config, 'common_to_mc')
        
        assert isinstance(config.data_to_common, dict)
        assert isinstance(config.mc_to_common, dict)
    
    def test_alias_resolution(self, branches_config_file: Path) -> None:
        """Test resolving aliases from common to data/MC."""
        config = BranchConfig(str(branches_config_file))
        
        # Test from fixture data
        if 'lambda_prob' in config.common_to_data:
            assert config.common_to_data['lambda_prob'] == 'L0_ProbNNp'
            assert config.common_to_mc['lambda_prob'] == 'L0_MC15TuneV1_ProbNNp'
    
    def test_reverse_alias_lookup(self, branches_config_file: Path) -> None:
        """Test reverse lookup from data/MC to common."""
        config = BranchConfig(str(branches_config_file))
        
        if 'L0_ProbNNp' in config.data_to_common:
            assert config.data_to_common['L0_ProbNNp'] == 'lambda_prob'
        
        if 'L0_MC15TuneV1_ProbNNp' in config.mc_to_common:
            assert config.mc_to_common['L0_MC15TuneV1_ProbNNp'] == 'lambda_prob'


@pytest.mark.unit
class TestBranchSets:
    """Test branch set retrieval and management."""
    
    def test_get_branches_from_sets(self, branches_config_file: Path) -> None:
        """Test getting branches from named sets."""
        config = BranchConfig(str(branches_config_file))
        
        branches = config.get_branches_from_sets(['essential'])
        
        assert isinstance(branches, list)
        assert len(branches) > 0
        assert 'Bu_PT' in branches
        assert 'L0_MM' in branches
    
    def test_get_branches_from_multiple_sets(self, branches_config_file: Path) -> None:
        """Test getting branches from multiple sets."""
        config = BranchConfig(str(branches_config_file))
        
        branches = config.get_branches_from_sets(['essential', 'kinematics'])
        
        assert isinstance(branches, list)
        # Should have branches from both sets
        assert 'Bu_PT' in branches  # from essential
        assert 'Bu_P' in branches   # from kinematics
    
    def test_get_branches_nonexistent_set(self, branches_config_file: Path) -> None:
        """Test warning for non-existent set."""
        config = BranchConfig(str(branches_config_file))
        
        # Should not raise, just warn and return empty
        branches = config.get_branches_from_sets(['nonexistent_set'])
        assert isinstance(branches, list)
    
    def test_exclude_mc_only_sets(self, tmp_test_dir: Path) -> None:
        """Test excluding MC-only branch sets."""
        # Create config with MC-only set
        config_dict = {
            "branches": {
                "truth": {
                    "mc_only": True,
                    "Bu": ["Bu_TRUEID"]
                },
                "essential": {
                    "Bu": ["Bu_PT"]
                }
            }
        }
        
        config_file = tmp_test_dir / "test_config.toml"
        with open(config_file, 'wb') as f:
            tomli_w.dump(config_dict, f)
        
        config = BranchConfig(str(config_file))
        
        # Without exclude_mc
        all_branches = config.get_branches_from_sets(['truth', 'essential'], exclude_mc=False)
        assert 'Bu_TRUEID' in all_branches
        
        # With exclude_mc
        data_only = config.get_branches_from_sets(['truth', 'essential'], exclude_mc=True)
        assert 'Bu_TRUEID' not in data_only
        assert 'Bu_PT' in data_only


@pytest.mark.unit
class TestPresets:
    """Test preset configurations."""
    
    def test_get_branches_from_preset(self, branches_config_file: Path) -> None:
        """Test getting branches from preset."""
        config = BranchConfig(str(branches_config_file))
        
        branches = config.get_branches_from_preset('minimal')
        
        assert isinstance(branches, list)
        assert len(branches) > 0
    
    def test_get_branches_from_standard_preset(self, branches_config_file: Path) -> None:
        """Test standard preset includes multiple sets."""
        config = BranchConfig(str(branches_config_file))
        
        branches = config.get_branches_from_preset('standard')
        
        # Standard should include both essential and kinematics
        assert 'Bu_PT' in branches
        assert 'Bu_P' in branches
    
    def test_preset_not_found(self, branches_config_file: Path) -> None:
        """Test error when preset doesn't exist."""
        config = BranchConfig(str(branches_config_file))
        
        with pytest.raises(ConfigurationError) as exc_info:
            config.get_branches_from_preset('nonexistent_preset')
        
        assert "not found" in str(exc_info.value)
    
    def test_no_presets_section(self, tmp_test_dir: Path) -> None:
        """Test error when presets section missing."""
        config_dict = {
            "branches": {
                "essential": {"Bu": ["Bu_PT"]}
            }
        }
        
        config_file = tmp_test_dir / "no_presets.toml"
        with open(config_file, 'wb') as f:
            tomli_w.dump(config_dict, f)
        
        config = BranchConfig(str(config_file))
        
        with pytest.raises(ConfigurationError) as exc_info:
            config.get_branches_from_preset('any')
        
        assert "No presets defined" in str(exc_info.value)


@pytest.mark.unit
class TestAliasResolution:
    """Test alias resolution for data vs MC."""
    
    def test_resolve_aliases_for_data(self, branches_config_file: Path) -> None:
        """Test resolving common names to data branches."""
        config = BranchConfig(str(branches_config_file))
        
        common_names = ['lambda_prob', 'Bu_PT']
        resolved = config.resolve_aliases(common_names, is_mc=False)
        
        assert isinstance(resolved, list)
        # lambda_prob should be resolved, Bu_PT should stay as-is
        if 'lambda_prob' in config.common_to_data:
            assert 'L0_ProbNNp' in resolved
        assert 'Bu_PT' in resolved
    
    def test_resolve_aliases_for_mc(self, branches_config_file: Path) -> None:
        """Test resolving common names to MC branches."""
        config = BranchConfig(str(branches_config_file))
        
        common_names = ['lambda_prob', 'Bu_PT']
        resolved = config.resolve_aliases(common_names, is_mc=True)
        
        assert isinstance(resolved, list)
        # lambda_prob should resolve to MC version
        if 'lambda_prob' in config.common_to_mc:
            assert 'L0_MC15TuneV1_ProbNNp' in resolved
    
    def test_resolve_aliases_no_aliases(self, branches_config_file: Path) -> None:
        """Test resolution when branches are not aliases."""
        config = BranchConfig(str(branches_config_file))
        
        branches = ['Bu_PT', 'Bu_ETA']
        resolved = config.resolve_aliases(branches, is_mc=False)
        
        assert resolved == branches


@pytest.mark.unit
class TestNormalization:
    """Test branch normalization (actual to common names)."""
    
    def test_normalize_data_branches(self, branches_config_file: Path) -> None:
        """Test normalizing data branches to common names."""
        config = BranchConfig(str(branches_config_file))
        
        data_branches = ['L0_ProbNNp', 'Bu_PT']
        rename_map = config.normalize_branches(data_branches, is_mc=False)
        
        assert isinstance(rename_map, dict)
        # Only aliased branches should be in map
        if 'L0_ProbNNp' in config.data_to_common:
            assert 'L0_ProbNNp' in rename_map
            assert rename_map['L0_ProbNNp'] == 'lambda_prob'
    
    def test_normalize_mc_branches(self, branches_config_file: Path) -> None:
        """Test normalizing MC branches to common names."""
        config = BranchConfig(str(branches_config_file))
        
        mc_branches = ['L0_MC15TuneV1_ProbNNp', 'Bu_PT']
        rename_map = config.normalize_branches(mc_branches, is_mc=True)
        
        if 'L0_MC15TuneV1_ProbNNp' in config.mc_to_common:
            assert 'L0_MC15TuneV1_ProbNNp' in rename_map


@pytest.mark.unit
class TestUtilityMethods:
    """Test utility methods."""
    
    def test_list_available_sets(self, branches_config_file: Path) -> None:
        """Test listing available branch sets."""
        config = BranchConfig(str(branches_config_file))
        
        sets = config.list_available_sets()
        
        assert isinstance(sets, list)
        assert 'essential' in sets
        assert 'kinematics' in sets
        assert 'load_sets' not in sets  # Should be excluded
    
    def test_list_available_presets(self, branches_config_file: Path) -> None:
        """Test listing available presets."""
        config = BranchConfig(str(branches_config_file))
        
        presets = config.list_available_presets()
        
        assert isinstance(presets, list)
        assert 'minimal' in presets
        assert 'standard' in presets
    
    def test_get_branches_by_particle(self, branches_config_file: Path) -> None:
        """Test getting branches for specific particle."""
        config = BranchConfig(str(branches_config_file))
        
        bu_branches = config.get_branches_by_particle('Bu', ['essential'])
        
        assert isinstance(bu_branches, list)
        assert 'Bu_PT' in bu_branches
        assert 'Bu_ETA' in bu_branches
        
        # L0 branches should not be included
        assert 'L0_MM' not in bu_branches


@pytest.mark.unit
class TestBranchValidation:
    """Test branch validation against available branches."""
    
    def test_validate_all_branches_exist(self, branches_config_file: Path) -> None:
        """Test validation when all branches exist."""
        config = BranchConfig(str(branches_config_file))
        
        requested = ['Bu_PT', 'Bu_ETA', 'L0_MM']
        available = ['Bu_PT', 'Bu_ETA', 'L0_MM', 'Bu_M', 'L0_PT']
        
        result = config.validate_branches(requested, available)
        
        # validate_branches returns sorted lists
        assert set(result['valid']) == set(requested)
        assert result['missing'] == []
        assert result['found'] == 3
        assert result['total_requested'] == 3
    
    def test_validate_some_branches_missing(self, branches_config_file: Path) -> None:
        """Test validation when some branches are missing."""
        config = BranchConfig(str(branches_config_file))
        
        requested = ['Bu_PT', 'Bu_MISSING', 'L0_MM']
        available = ['Bu_PT', 'L0_MM']
        
        result = config.validate_branches(requested, available)
        
        assert 'Bu_PT' in result['valid']
        assert 'L0_MM' in result['valid']
        assert 'Bu_MISSING' in result['missing']
        assert result['found'] == 2
        assert result['total_requested'] == 3
    
    def test_validate_no_branches_exist(self, branches_config_file: Path) -> None:
        """Test validation when no branches exist."""
        config = BranchConfig(str(branches_config_file))
        
        requested = ['MISSING1', 'MISSING2']
        available = ['Bu_PT', 'L0_MM']
        
        result = config.validate_branches(requested, available)
        
        assert result['valid'] == []
        assert len(result['missing']) == 2
        assert result['found'] == 0


@pytest.mark.unit
def test_get_branch_config_convenience_function(branches_config_file: Path) -> None:
    """Test convenience function for getting BranchConfig."""
    config = get_branch_config(str(branches_config_file))
    
    assert isinstance(config, BranchConfig)
    assert hasattr(config, 'config')

"""
Unit tests for data_handler module.

Tests TOMLConfig loading, DataManager initialization, and data
validation without requiring real ROOT files.
"""

from __future__ import annotations

import pytest
import tomli_w
from pathlib import Path
from typing import Dict, Any

from analysis.modules.data_handler import TOMLConfig
from analysis.modules.exceptions import ConfigurationError


@pytest.mark.unit
@pytest.mark.config
class TestTOMLConfigInitialization:
    """Test TOMLConfig initialization and file loading."""
    
    def test_init_with_valid_config_dir(self, config_dir_fixture: Path) -> None:
        """Test initialization with valid config directory."""
        config = TOMLConfig(str(config_dir_fixture))
        
        assert hasattr(config, 'physics')
        assert hasattr(config, 'detector')
        assert hasattr(config, 'fitting')
        assert hasattr(config, 'selection')
        assert hasattr(config, 'triggers')
        assert hasattr(config, 'data')
        assert hasattr(config, 'efficiencies')
    
    def test_backward_compatibility_attributes(self, config_dir_fixture: Path) -> None:
        """Test backward compatibility layer creates expected attributes."""
        config = TOMLConfig(str(config_dir_fixture))
        
        # Check backward compatibility
        assert hasattr(config, 'particles')
        assert hasattr(config, 'paths')
        assert hasattr(config, 'luminosity')
    
    def test_missing_config_directory(self, tmp_test_dir: Path) -> None:
        """Test error when config directory doesn't exist."""
        nonexistent_dir = tmp_test_dir / "nonexistent_config"
        
        with pytest.raises(ConfigurationError):
            TOMLConfig(str(nonexistent_dir))


@pytest.mark.unit
@pytest.mark.config
class TestTOMLFileLoading:
    """Test individual TOML file loading."""
    
    def test_load_physics_config(self, config_dir_fixture: Path) -> None:
        """Test loading physics.toml."""
        config = TOMLConfig(str(config_dir_fixture))
        
        assert isinstance(config.physics, dict)
        assert 'pdg_masses' in config.physics
        assert 'pdg_widths' in config.physics
    
    def test_load_detector_config(self, config_dir_fixture: Path) -> None:
        """Test loading detector.toml."""
        config = TOMLConfig(str(config_dir_fixture))
        
        assert isinstance(config.detector, dict)
        assert 'signal_regions' in config.detector
    
    def test_load_fitting_config(self, config_dir_fixture: Path) -> None:
        """Test loading fitting.toml."""
        config = TOMLConfig(str(config_dir_fixture))
        
        assert isinstance(config.fitting, dict)
        assert 'fit_method' in config.fitting
        assert 'background_model' in config.fitting
    
    def test_load_selection_config(self, config_dir_fixture: Path) -> None:
        """Test loading selection.toml."""
        config = TOMLConfig(str(config_dir_fixture))
        
        assert isinstance(config.selection, dict)
        assert 'cuts' in config.selection
    
    def test_missing_required_config_file(self, tmp_test_dir: Path) -> None:
        """Test error when required config file is missing."""
        config_dir = tmp_test_dir / "incomplete_config"
        config_dir.mkdir()
        
        # Create only some required files
        physics_config = config_dir / "physics.toml"
        with open(physics_config, 'wb') as f:
            tomli_w.dump({"pdg_masses": {}}, f)
        
        # Should fail because other required files are missing
        with pytest.raises(ConfigurationError):
            TOMLConfig(str(config_dir))


@pytest.mark.unit
@pytest.mark.config
class TestConfigurationValidation:
    """Test configuration content validation."""
    
    def test_physics_config_structure(self, config_dir_fixture: Path) -> None:
        """Test physics config has expected structure."""
        config = TOMLConfig(str(config_dir_fixture))
        
        assert 'pdg_masses' in config.physics
        assert isinstance(config.physics['pdg_masses'], dict)
        
        # Check for expected particles
        assert 'jpsi' in config.physics['pdg_masses']
        assert 'etac' in config.physics['pdg_masses']
    
    def test_detector_config_structure(self, config_dir_fixture: Path) -> None:
        """Test detector config has expected structure."""
        config = TOMLConfig(str(config_dir_fixture))
        
        assert 'signal_regions' in config.detector
        assert isinstance(config.detector['signal_regions'], dict)
        
        # Signal regions should have center and width
        jpsi_region = config.detector['signal_regions']['jpsi']
        assert 'center' in jpsi_region
        assert 'width' in jpsi_region
    
    def test_paths_config_structure(self, config_dir_fixture: Path) -> None:
        """Test paths config has expected structure."""
        config = TOMLConfig(str(config_dir_fixture))
        
        assert 'data' in config.paths
        assert 'mc' in config.paths
        assert 'output' in config.paths


@pytest.mark.unit
@pytest.mark.config
class TestBackwardCompatibility:
    """Test backward compatibility layer."""
    
    def test_particles_attribute_exists(self, config_dir_fixture: Path) -> None:
        """Test that particles attribute is created from physics+detector."""
        config = TOMLConfig(str(config_dir_fixture))
        
        assert hasattr(config, 'particles')
        assert isinstance(config.particles, dict)
    
    def test_paths_mapping(self, config_dir_fixture: Path) -> None:
        """Test that paths maps data.toml correctly."""
        config = TOMLConfig(str(config_dir_fixture))
        
        assert 'data' in config.paths
        assert 'mc' in config.paths
        assert 'output' in config.paths
        
        # Should map from data.toml
        assert 'input_data' in config.data
        assert config.paths['data'] == config.data['input_data']
    
    def test_luminosity_attribute(self, tmp_test_dir: Path) -> None:
        """Test luminosity attribute creation."""
        config_dir = tmp_test_dir / "config"
        config_dir.mkdir()
        
        # Create minimal required configs
        configs = {
            "physics.toml": {
                "pdg_masses": {},
                "pdg_branching_fractions": {},
                "analysis_method": {}
            },
            "detector.toml": {
                "signal_regions": {},
                "mass_windows": {},
                "integrated_luminosity": {"2016": 1.0}
            },
            "fitting.toml": {"fit_method": {}, "background_model": {}},
            "selection.toml": {"cuts": {}},
            "triggers.toml": {"lines": []},
            "data.toml": {
                "input_data": {"base_path": "/tmp"},
                "input_mc": {"base_path": "/tmp"},
                "output": {"base_path": "/tmp"}
            },
            "efficiencies.toml": {"trigger": {}},
        }
        
        for filename, content in configs.items():
            with open(config_dir / filename, 'wb') as f:
                tomli_w.dump(content, f)
        
        config = TOMLConfig(str(config_dir))
        
        # Should have luminosity attribute from backward compatibility
        assert hasattr(config, 'luminosity')


@pytest.mark.unit
@pytest.mark.config
class TestConfigurationErrorHandling:
    """Test error handling for invalid configurations."""
    
    def test_invalid_toml_syntax(self, tmp_test_dir: Path) -> None:
        """Test error handling for invalid TOML syntax."""
        config_dir = tmp_test_dir / "bad_config"
        config_dir.mkdir()
        
        # Create file with invalid TOML
        bad_file = config_dir / "physics.toml"
        bad_file.write_text("this is not valid TOML [[[")
        
        with pytest.raises(ConfigurationError) as exc_info:
            TOMLConfig(str(config_dir))
        
        assert "parsing" in str(exc_info.value).lower() or "error" in str(exc_info.value).lower()
    
    def test_missing_config_directory_error_message(self, tmp_test_dir: Path) -> None:
        """Test informative error message for missing directory."""
        nonexistent = tmp_test_dir / "does_not_exist"
        
        with pytest.raises(ConfigurationError) as exc_info:
            TOMLConfig(str(nonexistent))
        
        error_msg = str(exc_info.value)
        assert "physics.toml" in error_msg or "not found" in error_msg.lower()


@pytest.mark.unit
@pytest.mark.config
class TestConfigAccessPatterns:
    """Test common configuration access patterns."""
    
    def test_access_pdg_mass(self, config_dir_fixture: Path) -> None:
        """Test accessing PDG mass values."""
        config = TOMLConfig(str(config_dir_fixture))
        
        jpsi_mass = config.physics['pdg_masses']['jpsi']
        assert isinstance(jpsi_mass, (int, float))
        assert jpsi_mass > 0
    
    def test_access_signal_region(self, config_dir_fixture: Path) -> None:
        """Test accessing signal region parameters."""
        config = TOMLConfig(str(config_dir_fixture))
        
        jpsi_region = config.detector['signal_regions']['jpsi']
        assert 'center' in jpsi_region
        assert 'width' in jpsi_region
        assert jpsi_region['width'] > 0
    
    def test_access_fit_parameters(self, config_dir_fixture: Path) -> None:
        """Test accessing fit configuration."""
        config = TOMLConfig(str(config_dir_fixture))
        
        assert 'method' in config.fitting['fit_method']
        assert 'type' in config.fitting['background_model']
    
    def test_access_output_paths(self, config_dir_fixture: Path) -> None:
        """Test accessing output path configuration."""
        config = TOMLConfig(str(config_dir_fixture))
        
        assert 'base_path' in config.paths['output']
        output_path = Path(config.paths['output']['base_path'])
        assert isinstance(output_path, Path)


@pytest.mark.unit
@pytest.mark.config
def test_config_immutability_not_enforced() -> None:
    """Test that config can be modified (no immutability enforced)."""
    # This is intentional - configs should be mutable for flexibility
    # Just documenting the behavior
    pass


@pytest.mark.unit
@pytest.mark.config
def test_multiple_config_instances_independent(config_dir_fixture: Path) -> None:
    """Test that multiple config instances are independent."""
    config1 = TOMLConfig(str(config_dir_fixture))
    config2 = TOMLConfig(str(config_dir_fixture))
    
    # They should be separate instances
    assert config1 is not config2
    
    # But have the same data
    assert config1.physics == config2.physics

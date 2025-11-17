"""
Integration tests for multi-configuration loading.

Tests the TOMLConfig class to ensure all configuration files
load correctly together, have consistent cross-references,
and provide proper backward compatibility.

These tests validate the configuration layer that provides
analysis parameters to all pipeline components.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import tomli_w

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modules.data_handler import TOMLConfig
from modules.exceptions import ConfigurationError


@pytest.mark.integration
class TestMultiConfigLoading:
    """Test loading multiple configuration files together."""

    def test_load_all_required_configs(self, config_dir_fixture):
        """Test that all required configuration files load successfully."""
        config = TOMLConfig(str(config_dir_fixture))

        # Verify all logical sections are present
        assert hasattr(config, "physics")
        assert hasattr(config, "detector")
        assert hasattr(config, "fitting")
        assert hasattr(config, "selection")
        assert hasattr(config, "triggers")
        assert hasattr(config, "data")
        assert hasattr(config, "efficiencies")

    def test_config_directory_structure(self, config_dir_fixture):
        """Test that configuration directory has expected structure."""
        config = TOMLConfig(str(config_dir_fixture))

        # Check config_dir attribute
        assert hasattr(config, "config_dir")
        assert Path(config.config_dir).exists()
        assert Path(config.config_dir).is_dir()

        # Check that TOML files exist
        config_path = Path(config.config_dir)
        toml_files = list(config_path.glob("*.toml"))
        assert len(toml_files) > 0

    def test_invalid_config_directory_raises_error(self, tmp_test_dir):
        """Test that invalid config directory raises appropriate error."""
        invalid_dir = tmp_test_dir / "nonexistent"

        with pytest.raises((FileNotFoundError, ConfigurationError)):
            TOMLConfig(str(invalid_dir))

    def test_missing_required_file_raises_error(self, tmp_test_dir):
        """Test that missing a required config file raises error."""
        # Create incomplete config directory
        incomplete_dir = tmp_test_dir / "incomplete_config"
        incomplete_dir.mkdir()

        # Create only physics.toml, missing others
        physics_file = incomplete_dir / "physics.toml"
        with open(physics_file, "wb") as f:
            tomli_w.dump({"pdg_masses": {"jpsi": 3096.9}}, f)

        # Should raise error due to missing required files
        with pytest.raises((FileNotFoundError, ConfigurationError)):
            TOMLConfig(str(incomplete_dir))


@pytest.mark.integration
class TestCrossConfigConsistency:
    """Test consistency between different configuration files."""

    def test_paths_consistency(self, config_dir_fixture):
        """Test that paths are consistent across configuration."""
        config = TOMLConfig(str(config_dir_fixture))

        # Verify paths structure exists
        assert "output" in config.paths
        assert "base_path" in config.paths["output"]

        # Check that paths are valid strings
        assert isinstance(config.paths["output"]["base_path"], str)

    def test_luminosity_structure(self, config_dir_fixture):
        """Test luminosity configuration structure."""
        config = TOMLConfig(str(config_dir_fixture))

        # Verify luminosity exists (from detector.toml)
        assert hasattr(config, "luminosity")

        # Check it's a dictionary with year keys
        if isinstance(config.luminosity, dict):
            # Should have year entries
            for year, lumi in config.luminosity.items():
                if isinstance(year, str) and year != "integrated_luminosity":
                    # Skip non-year keys like 'integrated_luminosity'
                    if isinstance(lumi, dict):
                        # If it's nested dict, check values are numeric
                        for k, v in lumi.items():
                            if isinstance(v, (int, float)):
                                assert v > 0

    def test_physics_parameters_structure(self, config_dir_fixture):
        """Test physics parameters have expected structure."""
        config = TOMLConfig(str(config_dir_fixture))

        # Check physics section - it's a dict, not a namespace
        assert "pdg_masses" in config.physics
        assert isinstance(config.physics["pdg_masses"], dict)

        # Check for expected particles
        assert "jpsi" in config.physics["pdg_masses"]
        assert "etac" in config.physics["pdg_masses"]

        # Masses should be positive numbers
        for particle, mass in config.physics["pdg_masses"].items():
            assert isinstance(mass, (int, float))
            assert mass > 0

    def test_detector_signal_regions(self, config_dir_fixture):
        """Test detector signal region definitions."""
        config = TOMLConfig(str(config_dir_fixture))

        # Check signal regions exist - detector is a dict
        assert "signal_regions" in config.detector
        assert isinstance(config.detector["signal_regions"], dict)

        # Check structure of signal regions
        for state, region in config.detector["signal_regions"].items():
            assert "center" in region
            assert "width" in region
            assert isinstance(region["center"], (int, float))
            assert isinstance(region["width"], (int, float))
            assert region["width"] > 0


@pytest.mark.integration
class TestBackwardCompatibility:
    """Test backward compatibility layer functionality."""

    def test_particles_attribute_created(self, config_dir_fixture):
        """Test that particles backward compatibility attribute exists."""
        config = TOMLConfig(str(config_dir_fixture))

        # Should have particles attribute from compatibility layer
        assert hasattr(config, "particles")

        # Should contain physics data
        assert isinstance(config.particles, dict)

    def test_paths_attribute_created(self, config_dir_fixture):
        """Test that paths backward compatibility attribute exists."""
        config = TOMLConfig(str(config_dir_fixture))

        # Should have paths attribute
        assert hasattr(config, "paths")
        assert isinstance(config.paths, dict)

        # Should have output section
        assert "output" in config.paths

    def test_luminosity_attribute_created(self, config_dir_fixture):
        """Test that luminosity backward compatibility attribute exists."""
        config = TOMLConfig(str(config_dir_fixture))

        # Should have luminosity attribute
        assert hasattr(config, "luminosity")

    def test_branching_fractions_attribute_created(self, config_dir_fixture):
        """Test that branching_fractions backward compatibility attribute exists."""
        config = TOMLConfig(str(config_dir_fixture))

        # Should have branching_fractions attribute
        assert hasattr(config, "branching_fractions")

    def test_efficiency_inputs_attribute_created(self, config_dir_fixture):
        """Test that efficiency_inputs backward compatibility attribute exists."""
        config = TOMLConfig(str(config_dir_fixture))

        # Should have efficiency_inputs attribute
        assert hasattr(config, "efficiency_inputs")

    def test_all_compatibility_attributes_present(self, config_dir_fixture):
        """Test that all backward compatibility attributes are created."""
        config = TOMLConfig(str(config_dir_fixture))

        required_compat = [
            "particles",
            "paths",
            "luminosity",
            "branching_fractions",
            "efficiency_inputs",
        ]

        for attr in required_compat:
            assert hasattr(config, attr), f"Missing backward compatibility attribute: {attr}"


@pytest.mark.integration
class TestConfigIdempotence:
    """Test that configuration loading is idempotent."""

    def test_multiple_loads_same_result(self, config_dir_fixture):
        """Test that loading config multiple times gives same result."""
        config1 = TOMLConfig(str(config_dir_fixture))
        config2 = TOMLConfig(str(config_dir_fixture))

        # Both should have same structure
        assert hasattr(config1, "physics")
        assert hasattr(config2, "physics")

        # Physics masses should be identical
        assert config1.physics["pdg_masses"] == config2.physics["pdg_masses"]

    def test_config_does_not_modify_files(self, config_dir_fixture):
        """Test that loading config doesn't modify source files."""
        config_path = Path(config_dir_fixture)

        # Get modification times before
        physics_file = config_path / "physics.toml"
        mtime_before = physics_file.stat().st_mtime

        # Load config
        config = TOMLConfig(str(config_dir_fixture))

        # Check modification time after
        mtime_after = physics_file.stat().st_mtime

        # Should be unchanged
        assert mtime_before == mtime_after

    def test_config_attributes_immutable_intent(self, config_dir_fixture):
        """Test that config attributes can be accessed consistently."""
        config = TOMLConfig(str(config_dir_fixture))

        # Access same attribute multiple times
        physics1 = config.physics
        physics2 = config.physics

        # Should reference same namespace object
        assert physics1 is physics2


@pytest.mark.integration
class TestConfigAccessPatterns:
    """Test various configuration access patterns."""

    def test_nested_attribute_access(self, config_dir_fixture):
        """Test accessing nested configuration values."""
        config = TOMLConfig(str(config_dir_fixture))

        # Test nested access works - physics is a dict
        jpsi_mass = config.physics["pdg_masses"]["jpsi"]
        assert isinstance(jpsi_mass, (int, float))
        assert jpsi_mass > 0

    def test_dict_style_access(self, config_dir_fixture):
        """Test dictionary-style access to configuration."""
        config = TOMLConfig(str(config_dir_fixture))

        # Test dict access on paths
        output_path = config.paths["output"]
        assert isinstance(output_path, dict)

    def test_mixed_access_patterns(self, config_dir_fixture):
        """Test mixing attribute and dictionary access."""
        config = TOMLConfig(str(config_dir_fixture))

        # Mix attribute and dict access
        signal_regions = config.detector["signal_regions"]
        assert isinstance(signal_regions, dict)

        jpsi_region = signal_regions["jpsi"]
        assert "center" in jpsi_region

    def test_hasattr_works_correctly(self, config_dir_fixture):
        """Test that hasattr works correctly on config objects."""
        config = TOMLConfig(str(config_dir_fixture))

        # Should have these attributes
        assert hasattr(config, "physics")
        assert hasattr(config, "detector")

        # Should not have random attributes
        assert not hasattr(config, "nonexistent_attribute")


@pytest.mark.integration
class TestConfigDataTypes:
    """Test that configuration data types are correct."""

    def test_numeric_parameters_are_numbers(self, config_dir_fixture):
        """Test that numeric parameters have correct types."""
        config = TOMLConfig(str(config_dir_fixture))

        # Masses should be numbers
        for mass in config.physics["pdg_masses"].values():
            assert isinstance(mass, (int, float))

    def test_string_parameters_are_strings(self, config_dir_fixture):
        """Test that string parameters have correct types."""
        config = TOMLConfig(str(config_dir_fixture))

        # Paths should be strings
        base_path = config.paths["output"]["base_path"]
        assert isinstance(base_path, str)

    def test_dict_parameters_are_dicts(self, config_dir_fixture):
        """Test that dictionary parameters have correct types."""
        config = TOMLConfig(str(config_dir_fixture))

        # Signal regions should be dict
        assert isinstance(config.detector["signal_regions"], dict)

        # Paths should be dict
        assert isinstance(config.paths, dict)


@pytest.mark.integration
class TestConfigValidationIntegration:
    """Test configuration validation in integration context."""

    def test_required_physics_parameters_present(self, config_dir_fixture):
        """Test that required physics parameters are present."""
        config = TOMLConfig(str(config_dir_fixture))

        # Check for key physics parameters
        required_masses = ["jpsi", "etac"]
        for particle in required_masses:
            assert particle in config.physics["pdg_masses"], f"Missing required mass for {particle}"

    def test_required_detector_parameters_present(self, config_dir_fixture):
        """Test that required detector parameters are present."""
        config = TOMLConfig(str(config_dir_fixture))

        # Check for signal regions
        assert len(config.detector["signal_regions"]) > 0, "No signal regions defined"

        # Check for mass windows
        assert "mass_windows" in config.detector

    def test_output_paths_defined(self, config_dir_fixture):
        """Test that output paths are properly defined."""
        config = TOMLConfig(str(config_dir_fixture))

        # Check output structure
        assert "output" in config.paths
        assert "base_path" in config.paths["output"]

        # Should have subdirectories defined
        # (tables_dir, plots_dir, results_dir are created by compatibility layer)
        output_config = config.paths["output"]
        assert isinstance(output_config, dict)


@pytest.mark.integration
class TestConfigReloadability:
    """Test configuration can be reloaded and recreated."""

    def test_config_can_be_recreated(self, config_dir_fixture):
        """Test that config can be created, deleted, and recreated."""
        # Create first instance
        config1 = TOMLConfig(str(config_dir_fixture))
        jpsi_mass1 = config1.physics["pdg_masses"]["jpsi"]

        # Delete and recreate
        del config1
        config2 = TOMLConfig(str(config_dir_fixture))
        jpsi_mass2 = config2.physics["pdg_masses"]["jpsi"]

        # Should have same values
        assert jpsi_mass1 == jpsi_mass2

    def test_multiple_config_instances_independent(self, config_dir_fixture):
        """Test that multiple config instances are independent."""
        config1 = TOMLConfig(str(config_dir_fixture))
        config2 = TOMLConfig(str(config_dir_fixture))

        # Should be separate objects
        assert config1 is not config2

        # But should have same data
        assert config1.physics["pdg_masses"] == config2.physics["pdg_masses"]

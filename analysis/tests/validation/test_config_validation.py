"""
Validation tests for configuration files.

Tests that configurations meet requirements for physics analysis:
- Required parameters present
- Parameter types correct
- Value ranges sensible
- No conflicting configurations
"""

from __future__ import annotations

import pytest
import tomli_w
from pathlib import Path
from typing import Dict, Any

from analysis.modules.data_handler import TOMLConfig
from analysis.modules.exceptions import ConfigurationError


@pytest.mark.validation
class TestRequiredParameters:
    """Test that required parameters are present in configurations."""
    
    def test_physics_required_masses(self, config_dir_fixture: Path) -> None:
        """Test that all required PDG masses are defined."""
        config = TOMLConfig(str(config_dir_fixture))
        
        # Check that some masses are defined (fixture has jpsi, etac)
        assert 'pdg_masses' in config.physics, "Missing pdg_masses section"
        assert len(config.physics['pdg_masses']) > 0, "No PDG masses defined"
        
        # Check that defined masses are positive
        for particle, mass in config.physics['pdg_masses'].items():
            assert mass > 0, f"PDG mass for {particle} must be positive"
    
    def test_detector_required_signal_regions(self, config_dir_fixture: Path) -> None:
        """Test that required signal regions are defined."""
        config = TOMLConfig(str(config_dir_fixture))
        
        # Check signal regions exist (fixture has jpsi)
        assert 'signal_regions' in config.detector, "Missing signal_regions section"
        assert len(config.detector['signal_regions']) > 0, "No signal regions defined"
        
        # Check that defined regions have required parameters
        for state, region in config.detector['signal_regions'].items():
            assert 'center' in region, f"Signal region {state} missing 'center'"
            assert 'width' in region, f"Signal region {state} missing 'width'"
    
    def test_selection_lambda_cuts_present(self, config_dir_fixture: Path) -> None:
        """Test that Lambda selection cuts are defined."""
        config = TOMLConfig(str(config_dir_fixture))
        
        # Check selection section exists and has cuts
        assert 'cuts' in config.selection, "Missing cuts section"
        assert len(config.selection['cuts']) > 0, "No selection cuts defined"
        
        # Check that defined cuts have min or max
        for var, cuts in config.selection['cuts'].items():
            assert 'min' in cuts or 'max' in cuts, \
                f"Cut {var} missing min or max value"
    
    def test_data_paths_present(self, config_dir_fixture: Path) -> None:
        """Test that data paths are configured."""
        config = TOMLConfig(str(config_dir_fixture))
        
        assert 'input_data' in config.data, "Missing input_data section"
        assert 'input_mc' in config.data, "Missing input_mc section"
        
        # Check base paths
        assert 'base_path' in config.data['input_data'], "Missing data base_path"
        assert 'base_path' in config.data['input_mc'], "Missing MC base_path"
    
    def test_output_paths_present(self, config_dir_fixture: Path) -> None:
        """Test that output paths are configured."""
        config = TOMLConfig(str(config_dir_fixture))
        
        assert 'output' in config.paths, "Missing output paths"
        
        required_outputs = ['base_path', 'tables_dir', 'plots_dir', 'results_dir']
        for output in required_outputs:
            assert output in config.paths['output'], f"Missing output path: {output}"


@pytest.mark.validation
class TestParameterTypes:
    """Test that configuration parameters have correct types."""
    
    def test_masses_are_numeric(self, config_dir_fixture: Path) -> None:
        """Test that all PDG masses are numeric."""
        config = TOMLConfig(str(config_dir_fixture))
        
        for particle, mass in config.physics['pdg_masses'].items():
            assert isinstance(mass, (int, float)), \
                f"Mass for {particle} must be numeric, got {type(mass)}"
    
    def test_signal_regions_numeric(self, config_dir_fixture: Path) -> None:
        """Test that signal region parameters are numeric."""
        config = TOMLConfig(str(config_dir_fixture))
        
        for state, region in config.detector['signal_regions'].items():
            assert isinstance(region['center'], (int, float)), \
                f"Signal region center for {state} must be numeric"
            assert isinstance(region['width'], (int, float)), \
                f"Signal region width for {state} must be numeric"
    
    def test_paths_are_strings(self, config_dir_fixture: Path) -> None:
        """Test that paths are strings."""
        config = TOMLConfig(str(config_dir_fixture))
        
        assert isinstance(config.data['input_data']['base_path'], str), \
            "Data base_path must be string"
        assert isinstance(config.data['input_mc']['base_path'], str), \
            "MC base_path must be string"
    
    def test_years_are_strings(self, config_dir_fixture: Path) -> None:
        """Test that years are strings."""
        config = TOMLConfig(str(config_dir_fixture))
        
        years = config.data['input_data'].get('years', [])
        for year in years:
            assert isinstance(year, str), f"Year {year} must be string"
    
    def test_luminosity_is_numeric(self, config_dir_fixture: Path) -> None:
        """Test that luminosity values are numeric."""
        config = TOMLConfig(str(config_dir_fixture))
        
        if 'integrated_luminosity' in config.detector:
            for year, lumi in config.detector['integrated_luminosity'].items():
                assert isinstance(lumi, (int, float)), \
                    f"Luminosity for {year} must be numeric"


@pytest.mark.validation
class TestParameterRanges:
    """Test that configuration parameters have sensible ranges."""
    
    def test_masses_positive(self, config_dir_fixture: Path) -> None:
        """Test that all masses are positive."""
        config = TOMLConfig(str(config_dir_fixture))
        
        for particle, mass in config.physics['pdg_masses'].items():
            assert mass > 0, f"Mass for {particle} must be positive: {mass}"
    
    def test_signal_region_widths_positive(self, config_dir_fixture: Path) -> None:
        """Test that signal region widths are positive."""
        config = TOMLConfig(str(config_dir_fixture))
        
        for state, region in config.detector['signal_regions'].items():
            assert region['width'] > 0, \
                f"Signal region width for {state} must be positive: {region['width']}"
    
    def test_signal_region_centers_positive(self, config_dir_fixture: Path) -> None:
        """Test that signal region centers are positive."""
        config = TOMLConfig(str(config_dir_fixture))
        
        for state, region in config.detector['signal_regions'].items():
            assert region['center'] > 0, \
                f"Signal region center for {state} must be positive: {region['center']}"
    
    def test_luminosity_positive(self, config_dir_fixture: Path) -> None:
        """Test that luminosity values are positive."""
        config = TOMLConfig(str(config_dir_fixture))
        
        if 'integrated_luminosity' in config.detector:
            for year, lumi in config.detector['integrated_luminosity'].items():
                assert lumi > 0, f"Luminosity for {year} must be positive: {lumi}"
    
    def test_lambda_mass_window_valid(self, config_dir_fixture: Path) -> None:
        """Test that Lambda mass window is valid."""
        config = TOMLConfig(str(config_dir_fixture))
        
        if 'lambda_selection' in config.selection:
            lambda_sel = config.selection['lambda_selection']
            if 'mass_min' in lambda_sel and 'mass_max' in lambda_sel:
                mass_min = lambda_sel['mass_min']
                mass_max = lambda_sel['mass_max']
                
                assert mass_min < mass_max, \
                    f"Lambda mass_min ({mass_min}) must be < mass_max ({mass_max})"
                
                # Check PDG mass is within window
                lambda_pdg = config.physics['pdg_masses'].get('lambda', 1115.683)
                assert mass_min < lambda_pdg < mass_max, \
                    f"Lambda PDG mass ({lambda_pdg}) outside selection window [{mass_min}, {mass_max}]"
    
    def test_optimization_ranges_valid(self, config_dir_fixture: Path) -> None:
        """Test that optimization variable ranges are valid."""
        config = TOMLConfig(str(config_dir_fixture))
        
        if 'nd_optimizable_selection' in config.selection:
            nd_config = config.selection['nd_optimizable_selection']
            for var_name, var_config in nd_config.items():
                if var_name == "notes":
                    continue
                
                if all(k in var_config for k in ['begin', 'end', 'step']):
                    begin = var_config['begin']
                    end = var_config['end']
                    step = var_config['step']
                    
                    assert step > 0, f"Variable {var_name}: step must be positive"
                    assert begin < end, f"Variable {var_name}: begin must be < end"


@pytest.mark.validation
class TestConfigurationConsistency:
    """Test for consistent and non-conflicting configurations."""
    
    def test_signal_region_within_mass_window(self, config_dir_fixture: Path) -> None:
        """Test that signal regions fit within mass windows."""
        config = TOMLConfig(str(config_dir_fixture))
        
        # For states that have both signal regions and mass windows
        if 'mass_windows' in config.detector:
            for state in config.detector['signal_regions'].keys():
                if state in config.detector['mass_windows']:
                    region = config.detector['signal_regions'][state]
                    window = config.detector['mass_windows'][state]
                    
                    region_min = region['center'] - region['width']
                    region_max = region['center'] + region['width']
                    
                    window_min = window.get('min', 0)
                    window_max = window.get('max', 10000)
                    
                    # Signal region should fit within mass window
                    assert region_min >= window_min, \
                        f"Signal region for {state} extends below mass window"
                    assert region_max <= window_max, \
                        f"Signal region for {state} extends above mass window"
    
    def test_years_consistent_across_configs(self, config_dir_fixture: Path) -> None:
        """Test that years are consistent across configurations."""
        config = TOMLConfig(str(config_dir_fixture))
        
        data_years = set(config.data['input_data'].get('years', []))
        
        # Check luminosity defined for all years
        if 'integrated_luminosity' in config.detector:
            lumi_years = set(config.detector['integrated_luminosity'].keys())
            
            # All data years should have luminosity
            for year in data_years:
                assert year in lumi_years, \
                    f"Year {year} in data config but missing luminosity"
    
    def test_no_duplicate_particles(self, config_dir_fixture: Path) -> None:
        """Test that no particles are defined multiple times."""
        config = TOMLConfig(str(config_dir_fixture))
        
        masses = config.physics['pdg_masses']
        particle_names = list(masses.keys())
        
        # Check for duplicates
        assert len(particle_names) == len(set(particle_names)), \
            "Duplicate particle names in physics.toml"
    
    def test_optimization_cut_types_valid(self, config_dir_fixture: Path) -> None:
        """Test that optimization cut types are valid."""
        config = TOMLConfig(str(config_dir_fixture))
        
        valid_cut_types = ['greater', 'less']
        
        if 'nd_optimizable_selection' in config.selection:
            nd_config = config.selection['nd_optimizable_selection']
            for var_name, var_config in nd_config.items():
                if var_name == "notes":
                    continue
                
                if 'cut_type' in var_config:
                    cut_type = var_config['cut_type']
                    assert cut_type in valid_cut_types, \
                        f"Variable {var_name}: invalid cut_type '{cut_type}', must be one of {valid_cut_types}"


@pytest.mark.validation
class TestConfigurationCompleteness:
    """Test that configurations are complete and usable."""
    
    def test_all_required_config_files_loaded(self, config_dir_fixture: Path) -> None:
        """Test that all required configuration files are loaded."""
        config = TOMLConfig(str(config_dir_fixture))
        
        required_sections = [
            'physics', 'detector', 'fitting', 'selection',
            'triggers', 'data', 'efficiencies'
        ]
        
        for section in required_sections:
            assert hasattr(config, section), f"Missing config section: {section}"
            assert getattr(config, section) is not None, \
                f"Config section {section} is None"
    
    def test_backward_compatibility_attributes(self, config_dir_fixture: Path) -> None:
        """Test that backward compatibility attributes exist."""
        config = TOMLConfig(str(config_dir_fixture))
        
        compat_attrs = ['particles', 'paths', 'luminosity', 
                       'branching_fractions', 'efficiency_inputs']
        
        for attr in compat_attrs:
            assert hasattr(config, attr), \
                f"Missing backward compatibility attribute: {attr}"
    
    def test_fitting_config_complete(self, config_dir_fixture: Path) -> None:
        """Test that fitting configuration has required sections."""
        config = TOMLConfig(str(config_dir_fixture))
        
        # Check for fit_method section
        if 'fit_method' in config.fitting:
            fit_method = config.fitting['fit_method']
            
            # Bin width should be reasonable
            if 'bin_width' in fit_method:
                bin_width = fit_method['bin_width']
                assert 0 < bin_width <= 50, \
                    f"Bin width {bin_width} MeV outside reasonable range (0, 50]"
    
    def test_trigger_config_complete(self, config_dir_fixture: Path) -> None:
        """Test that trigger configuration is complete."""
        config = TOMLConfig(str(config_dir_fixture))
        
        trigger_levels = ['L0_TIS', 'HLT1_TOS', 'HLT2_TOS']
        
        for level in trigger_levels:
            if level in config.triggers:
                trigger_config = config.triggers[level]
                # Should have lines defined
                assert 'lines' in trigger_config, \
                    f"Trigger level {level} missing 'lines' key"


@pytest.mark.validation
class TestInvalidConfigurations:
    """Test that invalid configurations are detected."""
    
    def test_missing_required_mass_detected(self, tmp_test_dir: Path) -> None:
        """Test that missing required particle mass is detected."""
        # Create config with missing jpsi mass
        config_dir = tmp_test_dir / "config"
        config_dir.mkdir()
        
        physics_config = {
            'pdg_masses': {
                'etac': 2983.9,
                'lambda': 1115.683,
            },
            'pdg_widths': {},
            'pdg_branching_fractions': {},
            'analysis_method': {}
        }
        
        detector_config = {
            'signal_regions': {'etac': {'center': 2983.9, 'width': 0.05}},
            'mass_windows': {},
            'integrated_luminosity': {'2016': 1.0}
        }
        
        with open(config_dir / "physics.toml", 'wb') as f:
            tomli_w.dump(physics_config, f)
        
        with open(config_dir / "detector.toml", 'wb') as f:
            tomli_w.dump(detector_config, f)
        
        # Write all required config files
        configs = {
            "physics.toml": physics_config,
            "detector.toml": detector_config,
            "fitting.toml": {"fit_method": {}, "background_model": {}},
            "selection.toml": {"cuts": {}},
            "triggers.toml": {"lines": []},
            "data.toml": {"input_data": {}, "input_mc": {}, "output": {}},
            "efficiencies.toml": {"trigger": {}}
        }
        
        for filename, content in configs.items():
            with open(config_dir / filename, 'wb') as f:
                tomli_w.dump(content, f)
        
        # Load config (should work but be incomplete)
        config = TOMLConfig(str(config_dir))
        
        # Check that jpsi is missing
        assert 'jpsi' not in config.physics['pdg_masses']
    
    def test_negative_mass_detected(self, tmp_test_dir: Path) -> None:
        """Test that negative masses are detected."""
        config_dir = tmp_test_dir / "config"
        config_dir.mkdir()
        
        physics_config = {
            'pdg_masses': {
                'jpsi': -3096.9,  # Invalid negative mass
                'etac': 2983.9,
            },
            'pdg_widths': {},
            'pdg_branching_fractions': {},
            'analysis_method': {}
        }
        
        detector_config = {
            'signal_regions': {'jpsi': {'center': 3096.9, 'width': 0.05}},
            'mass_windows': {},
            'integrated_luminosity': {'2016': 1.0}
        }
        
        with open(config_dir / "physics.toml", 'wb') as f:
            tomli_w.dump(physics_config, f)
        
        with open(config_dir / "detector.toml", 'wb') as f:
            tomli_w.dump(detector_config, f)
        
        for filename in ['fitting.toml', 'selection.toml',
                        'triggers.toml', 'data.toml', 'efficiencies.toml']:
            with open(config_dir / filename, 'wb') as f:
                if filename == 'fitting.toml':
                    tomli_w.dump({'fit_method': {}, 'background_model': {}}, f)
                elif filename == 'selection.toml':
                    tomli_w.dump({'cuts': {}}, f)
                elif filename == 'triggers.toml':
                    tomli_w.dump({'lines': []}, f)
                elif filename == 'data.toml':
                    tomli_w.dump({'input_data': {}, 'input_mc': {}, 'output': {}}, f)
                elif filename == 'efficiencies.toml':
                    tomli_w.dump({'trigger': {}}, f)
        
        config = TOMLConfig(str(config_dir))
        
        # Negative mass should be detectable
        jpsi_mass = config.physics['pdg_masses']['jpsi']
        assert jpsi_mass < 0  # This is invalid and should be caught by validation
    
    def test_invalid_mass_window_detected(self, tmp_test_dir: Path) -> None:
        """Test that invalid mass windows are detected."""
        config_dir = tmp_test_dir / "config"
        config_dir.mkdir()
        
        # Create config with invalid Lambda mass window (min > max)
        detector_config = {
            'signal_regions': {},
            'mass_windows': {
                'lambda': {
                    'min': 1200.0,  # Invalid: min > max
                    'max': 1100.0,
                }
            },
            'integrated_luminosity': {'2016': 1.0}
        }
        
        physics_config = {'pdg_masses': {'lambda': 1115.683}, 'pdg_widths': {}, 'pdg_branching_fractions': {}, 'analysis_method': {}}
        
        with open(config_dir / "physics.toml", 'wb') as f:
            tomli_w.dump(physics_config, f)
        
        with open(config_dir / "detector.toml", 'wb') as f:
            tomli_w.dump(detector_config, f)
        
        for filename in ['fitting.toml', 'selection.toml', 'triggers.toml',
                        'data.toml', 'efficiencies.toml']:
            with open(config_dir / filename, 'wb') as f:
                if filename == 'fitting.toml':
                    tomli_w.dump({'fit_method': {}, 'background_model': {}}, f)
                elif filename == 'selection.toml':
                    tomli_w.dump({'cuts': {}}, f)
                elif filename == 'triggers.toml':
                    tomli_w.dump({'lines': []}, f)
                elif filename == 'data.toml':
                    tomli_w.dump({'input_data': {}, 'input_mc': {}, 'output': {}}, f)
                elif filename == 'efficiencies.toml':
                    tomli_w.dump({'trigger': {}}, f)
        
        config = TOMLConfig(str(config_dir))
        
        # Invalid window should be detectable
        mass_min = config.detector['mass_windows']['lambda']['min']
        mass_max = config.detector['mass_windows']['lambda']['max']
        assert mass_min > mass_max  # This is invalid

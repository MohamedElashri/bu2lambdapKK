#!/usr/bin/env python3
"""
Configuration Validation Script

Validates all configuration files before running the pipeline.
Catches errors early instead of 20 minutes into a run.

Usage:
    python validate_config.py
    python validate_config.py --verbose
"""

import argparse
import sys
from pathlib import Path

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.data_handler import TOMLConfig
from modules.exceptions import ConfigurationError


class ConfigValidator:
    """
    Validates configuration files for the Bu2lambdapKK analysis pipeline

    Checks:
    - All required config files exist
    - All required fields are present
    - Value ranges are sensible
    - File paths exist
    - No conflicting parameters
    """

    def __init__(self, config_dir: str = "config", verbose: bool = False):
        """
        Initialize validator

        Args:
            config_dir: Path to config directory
            verbose: Print detailed validation info
        """
        self.config_dir = Path(config_dir)
        self.verbose = verbose
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def validate_all(self) -> bool:
        """
        Run all validation checks

        Returns:
            bool: True if all checks pass, False otherwise
        """
        print("=" * 80)
        print("CONFIGURATION VALIDATION")
        print("=" * 80)

        try:
            # Load configuration
            print("\n1. Loading configuration files...")
            self.config = TOMLConfig(str(self.config_dir))
            print("   [OK] All config files loaded")

            # Run validation checks
            checks = [
                ("File paths", self._validate_file_paths),
                ("Physics constants", self._validate_physics),
                ("Detector parameters", self._validate_detector),
                ("Fitting configuration", self._validate_fitting),
                ("Selection parameters", self._validate_selection),
                ("Trigger configuration", self._validate_triggers),
                ("Efficiency inputs", self._validate_efficiencies),
            ]

            for check_name, check_func in checks:
                print(f"\n2. Validating {check_name}...")
                check_func()
                if not self.errors:
                    print(f"   [OK] {check_name} valid")

            # Print summary
            self._print_summary()

            return len(self.errors) == 0

        except ConfigurationError as e:
            print(f"\n[ERROR] Configuration loading failed: {e}")
            return False
        except Exception as e:
            print(f"\n[ERROR] Unexpected error during validation: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _validate_file_paths(self):
        """Validate that data file paths exist"""
        # Check data path
        data_path = Path(self.config.data["input_data"]["base_path"])
        if not data_path.exists():
            self.warnings.append(f"Data path does not exist: {data_path}")
            self.warnings.append("  → Data files must be present before running pipeline")
        elif self.verbose:
            print(f"   Data path: {data_path}")

        # Check MC path
        mc_path = Path(self.config.data["input_mc"]["base_path"])
        if not mc_path.exists():
            self.warnings.append(f"MC path does not exist: {mc_path}")
            self.warnings.append("  → MC files must be present before running pipeline")
        elif self.verbose:
            print(f"   MC path: {mc_path}")

        # Check years and magnets are specified
        years = self.config.data["input_data"].get("years", [])
        if not years:
            self.errors.append("No years specified in data.toml")
        elif len(years) == 0:
            self.errors.append("Years list is empty in data.toml")
        elif self.verbose:
            print(f"   Years: {', '.join(years)}")

        magnets = self.config.data["input_data"].get("magnets", [])
        if not magnets or len(magnets) == 0:
            self.errors.append("No magnets specified in data.toml")
        elif self.verbose:
            print(f"   Magnets: {', '.join(magnets)}")

    def _validate_physics(self):
        """Validate physics constants"""
        # Check PDG masses
        required_masses = [
            "jpsi",
            "etac_1s",
            "chic0",
            "chic1",
            "etac_2s",
            "lambda",
            "proton",
            "kaon",
        ]
        for particle in required_masses:
            if particle not in self.config.physics["pdg_masses"]:
                self.errors.append(f"Missing PDG mass for {particle} in physics.toml")
            else:
                mass = self.config.physics["pdg_masses"][particle]
                if mass <= 0:
                    self.errors.append(f"Invalid mass for {particle}: {mass} (must be > 0)")
                elif self.verbose:
                    print(f"   {particle}: {mass} MeV/c²")

        # Check PDG widths
        if "pdg_widths" in self.config.physics:
            for state, width in self.config.physics["pdg_widths"].items():
                if width < 0:
                    self.errors.append(f"Invalid width for {state}: {width} (must be >= 0)")

    def _validate_detector(self):
        """Validate detector parameters"""
        # Check signal regions
        required_states = ["jpsi", "etac", "chic0", "chic1", "etac_2s"]
        for state in required_states:
            if state not in self.config.detector["signal_regions"]:
                self.errors.append(f"Missing signal region for {state} in detector.toml")
            else:
                region = self.config.detector["signal_regions"][state]
                if "center" not in region or "window" not in region:
                    self.errors.append(f"Signal region for {state} missing 'center' or 'window'")
                else:
                    center = region["center"]
                    window = region["window"]
                    if center <= 0:
                        self.errors.append(f"Invalid center for {state}: {center}")
                    if window <= 0:
                        self.errors.append(f"Invalid window for {state}: {window}")
                    elif self.verbose:
                        print(f"   {state}: {center} ± {window} MeV/c²")

        # Check luminosity
        years = self.config.data["input_data"].get("years", [])
        for year in years:
            if year not in self.config.detector["integrated_luminosity"]:
                self.errors.append(f"Missing luminosity for year {year} in detector.toml")
            else:
                lumi = self.config.detector["integrated_luminosity"][year]
                if lumi <= 0:
                    self.errors.append(f"Invalid luminosity for {year}: {lumi}")
                elif self.verbose:
                    print(f"   {year}: {lumi} fb⁻¹")

    def _validate_fitting(self):
        """Validate fitting configuration"""
        # Check fit method
        if "fit_method" not in self.config.fitting:
            self.errors.append("Missing [fit_method] section in fitting.toml")
            return

        fit_method = self.config.fitting["fit_method"]

        # Check binning
        if "bin_width" in fit_method:
            bin_width = fit_method["bin_width"]
            if bin_width <= 0:
                self.errors.append(f"Invalid bin_width: {bin_width} (must be > 0)")
            elif bin_width > 50:
                self.warnings.append(f"Large bin_width: {bin_width} MeV (typical: 5-10 MeV)")
            elif self.verbose:
                print(f"   Bin width: {bin_width} MeV")

        # Check ARGUS parameters
        if "background_model" in self.config.fitting:
            bg_model = self.config.fitting["background_model"]
            if "argus_endpoint_offset" in bg_model:
                offset = bg_model["argus_endpoint_offset"]
                if offset < 0:
                    self.errors.append(f"Invalid ARGUS offset: {offset} (must be >= 0)")
                elif offset > 500:
                    self.warnings.append(f"Large ARGUS offset: {offset} MeV (typical: 100-300 MeV)")
                elif self.verbose:
                    print(f"   ARGUS endpoint offset: {offset} MeV")

    def _validate_selection(self):
        """Validate selection parameters"""
        # Check Lambda cuts
        if "lambda_selection" not in self.config.selection:
            self.errors.append("Missing [lambda_selection] section in selection.toml")
            return

        lambda_cuts = self.config.selection["lambda_selection"]

        # Check mass window
        if "mass_min" in lambda_cuts and "mass_max" in lambda_cuts:
            mass_min = lambda_cuts["mass_min"]
            mass_max = lambda_cuts["mass_max"]
            if mass_min >= mass_max:
                self.errors.append(f"Invalid Lambda mass window: [{mass_min}, {mass_max}]")
            lambda_pdg = self.config.physics["pdg_masses"].get("lambda", 1115.683)
            if not (mass_min < lambda_pdg < mass_max):
                self.warnings.append(f"Lambda PDG mass ({lambda_pdg}) outside selection window")

        # Check N-D optimization variables
        if "nd_optimizable_selection" not in self.config.selection:
            self.errors.append("Missing [nd_optimizable_selection] section in selection.toml")
            return

        nd_config = self.config.selection["nd_optimizable_selection"]
        var_count = 0
        for var_name, var_config in nd_config.items():
            if var_name == "notes":
                continue
            var_count += 1

            # Check required fields
            required = ["begin", "end", "step", "cut_type", "branch_name"]
            for field in required:
                if field not in var_config:
                    self.errors.append(f"Variable {var_name} missing field '{field}'")

            # Check value ranges
            if "begin" in var_config and "end" in var_config and "step" in var_config:
                begin = var_config["begin"]
                end = var_config["end"]
                step = var_config["step"]

                if step <= 0:
                    self.errors.append(f"Variable {var_name}: step must be > 0")
                if begin >= end:
                    self.errors.append(f"Variable {var_name}: begin >= end")

                n_points = int((end - begin) / step) + 1
                if n_points > 20:
                    self.warnings.append(f"Variable {var_name}: {n_points} points (may be slow)")
                elif self.verbose:
                    print(f"   {var_name}: {n_points} scan points")

            # Check cut type
            if "cut_type" in var_config:
                cut_type = var_config["cut_type"]
                if cut_type not in ["greater", "less"]:
                    self.errors.append(f"Variable {var_name}: invalid cut_type '{cut_type}'")

        if var_count == 0:
            self.errors.append("No variables defined in nd_optimizable_selection")
        elif self.verbose:
            print(f"   Total optimization variables: {var_count}")

        # Check sideband multipliers
        if "optimization_strategy" in self.config.selection:
            opt_strat = self.config.selection["optimization_strategy"]
            sideband_params = [
                "sideband_low_multiplier",
                "sideband_low_end_multiplier",
                "sideband_high_start_multiplier",
                "sideband_high_multiplier",
            ]
            for param in sideband_params:
                if param in opt_strat:
                    value = opt_strat[param]
                    if value < 0:
                        self.errors.append(f"Invalid {param}: {value} (must be >= 0)")

    def _validate_triggers(self):
        """Validate trigger configuration"""
        trigger_levels = ["L0_TIS", "HLT1_TOS", "HLT2_TOS"]
        for level in trigger_levels:
            if level not in self.config.triggers:
                self.errors.append(f"Missing [{level}] section in triggers.toml")
            else:
                lines = self.config.triggers[level].get("lines", [])
                if not lines or len(lines) == 0:
                    self.warnings.append(f"No trigger lines defined for {level}")
                elif self.verbose:
                    print(f"   {level}: {len(lines)} lines")

    def _validate_efficiencies(self):
        """Validate efficiency inputs"""
        # Just check that sections exist
        required_sections = ["efficiency_components", "stripping_efficiency", "trigger_efficiency"]
        for section in required_sections:
            if section not in self.config.efficiencies:
                self.warnings.append(f"Missing [{section}] section in efficiencies.toml")

        # Check that efficiency values are in valid range [0, 1]
        if "stripping_efficiency" in self.config.efficiencies:
            for state, years in self.config.efficiencies["stripping_efficiency"].items():
                if isinstance(years, dict):
                    for year, eff_dict in years.items():
                        if isinstance(eff_dict, dict) and "value" in eff_dict:
                            value = eff_dict["value"]
                            if not (0 <= value <= 1):
                                self.warnings.append(
                                    f"Efficiency for {state} {year}: {value} "
                                    f"(expected range [0, 1])"
                                )

    def _print_summary(self):
        """Print validation summary"""
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)

        if self.errors:
            print(f"\n[ERRORS] Found {len(self.errors)} error(s):")
            for error in self.errors:
                print(f"  - {error}")

        if self.warnings:
            print(f"\n[WARNINGS] Found {len(self.warnings)} warning(s):")
            for warning in self.warnings:
                print(f"  - {warning}")

        if not self.errors and not self.warnings:
            print("\n[SUCCESS] All checks passed!")
            print("Configuration is valid. Ready to run pipeline.")
        elif not self.errors:
            print(f"\n[SUCCESS] No errors (but {len(self.warnings)} warnings)")
            print("Warnings are informational. Pipeline can run.")
        else:
            print(f"\n[FAILED] Validation failed: {len(self.errors)} error(s)")
            print("Please fix errors before running pipeline.")

        print("=" * 80)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Validate configuration files for Bu2lambdapKK analysis"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print detailed validation info"
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="config",
        help="Path to config directory (default: config)",
    )

    args = parser.parse_args()

    # Run validation
    validator = ConfigValidator(config_dir=args.config_dir, verbose=args.verbose)

    success = validator.validate_all()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

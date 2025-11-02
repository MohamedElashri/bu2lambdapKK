import uproot
from pathlib import Path
from typing import Dict, Set
import json

class BranchInspector:
    """
    Inspect ROOT files and document available branches
    Creates markdown documentation of branch structure
    """
    
    def __init__(self, data_path: str, mc_path: str):
        self.data_path = Path(data_path)
        self.mc_path = Path(mc_path)
        
    def get_branches_from_file(self, filepath: Path, tree_name: str = "DecayTree") -> Set[str]:
        """Extract all branch names from a ROOT file"""
        with uproot.open(filepath) as file:
            tree = file[tree_name]
            return set(tree.keys())
    
    def inspect_data_files(self) -> Dict[str, Set[str]]:
        """
        Inspect all data files and check for branch consistency
        
        Returns:
            Dictionary: {year_magnet: set_of_branches}
        """
        data_branches = {}
        
        for year in [2016, 2017, 2018]:
            for magnet in ["MD", "MU"]:
                filename = f"dataBu2L0barPHH_{year-2000}{magnet}.root"
                filepath = self.data_path / filename
                
                if filepath.exists():
                    branches = self.get_branches_from_file(filepath)
                    key = f"{year}_{magnet}"
                    data_branches[key] = branches
                    print(f"✓ Data {key}: {len(branches)} branches")
        
        return data_branches
    
    def inspect_mc_files(self) -> Dict[str, Dict[str, Set[str]]]:
        """
        Inspect all MC files for each particle type
        
        Returns:
            Nested dict: {particle_type: {year_magnet: set_of_branches}}
        """
        mc_branches = {}
        
        for particle in ["Jpsi", "etac", "chic0", "chic1", "KpKm"]:
            mc_branches[particle] = {}
            
            for year in [2016, 2017, 2018]:
                for magnet in ["MD", "MU"]:
                    filename = f"{particle}_{year-2000}_{magnet}.root"
                    filepath = self.mc_path / particle / filename
                    
                    if filepath.exists():
                        branches = self.get_branches_from_file(filepath)
                        key = f"{year}_{magnet}"
                        mc_branches[particle][key] = branches
                        print(f"✓ MC {particle} {key}: {len(branches)} branches")
        
        return mc_branches
    
    def find_common_and_unique_branches(self, 
                                       data_branches: Dict[str, Set[str]],
                                       mc_branches: Dict[str, Dict[str, Set[str]]]) -> Dict:
        """
        Identify:
        - Branches common to ALL files (data + all MC)
        - Branches only in data
        - Branches only in MC (e.g., truth-matching)
        - Branches that differ between MC samples
        """
        # Flatten MC structure
        all_mc_branches = set()
        for particle in mc_branches:
            for year_mag in mc_branches[particle]:
                all_mc_branches.update(mc_branches[particle][year_mag])
        
        # Flatten data structure
        all_data_branches = set()
        for year_mag in data_branches:
            all_data_branches.update(data_branches[year_mag])
        
        common = all_data_branches & all_mc_branches
        data_only = all_data_branches - all_mc_branches
        mc_only = all_mc_branches - all_data_branches
        
        return {
            "common": sorted(common),
            "data_only": sorted(data_only),
            "mc_only": sorted(mc_only)
        }
    
    def identify_required_branches(self) -> Dict[str, str]:
        """
        Based on analysis needs, identify critical branches
        
        Returns:
            Dictionary mapping logical name to actual branch name(s)
            
        Example:
            {
                "bu_mass": "Bu_MM",
                "lambda_mass": "L0_MM",
                "kplus_px": "Kplus_PX"  or "h1_PX"  ← Need to discover!
            }
        """
        # This will be populated after inspection
        required = {
            # B+ properties
            "bu_mass": "Bu_MM",
            "bu_mass_corrected": "DERIVED",  # Bu_MM - L0_MM + 1115.683
            "bu_pt": "Bu_PT",
            "bu_dtf_chi2": "Bu_DTF_chi2",
            "bu_ipchi2": "Bu_IPCHI2_OWNPV",
            "bu_fdchi2": "Bu_FDCHI2_OWNPV",
            "bu_endvertex_x": "Bu_ENDVERTEX_X",
            "bu_endvertex_y": "Bu_ENDVERTEX_Y",
            "bu_endvertex_z": "Bu_ENDVERTEX_Z",
            "bu_endvertex_xerr": "Bu_ENDVERTEX_XERR",
            "bu_endvertex_yerr": "Bu_ENDVERTEX_YERR",
            "bu_endvertex_zerr": "Bu_ENDVERTEX_ZERR",
            
            # Lambda properties
            "lambda_mass": "L0_MM",
            "lambda_fdchi2": "L0_FD_CHISQ",
            "lambda_endvertex_x": "L0_ENDVERTEX_X",
            "lambda_endvertex_y": "L0_ENDVERTEX_Y",
            "lambda_endvertex_z": "L0_ENDVERTEX_Z",
            "lambda_endvertex_xerr": "L0_ENDVERTEX_XERR",
            "lambda_endvertex_yerr": "L0_ENDVERTEX_YERR",
            "lambda_endvertex_zerr": "L0_ENDVERTEX_ZERR",
            
            # Lambda daughters (proton from Lambda)
            "lambda_proton_probnnp": "Lp_ProbNNp",
            
            # Invariant mass of Λ̄pK⁻ (charmonium candidate)
            "charmonium_mass": "NEEDS_CONSTRUCTION",  # From 4-momenta
            
            # Four-momenta for mass reconstruction
            # ← NEED TO DISCOVER ACTUAL BRANCH NAMES
            "lambda_px": "L0_PX",  # ?
            "lambda_py": "L0_PY",  # ?
            "lambda_pz": "L0_PZ",  # ?
            "lambda_pe": "L0_PE",  # ?
            
            "bachelor_p_px": "???",  # Bachelor antiproton
            "bachelor_p_py": "???",
            "bachelor_p_pz": "???",
            "bachelor_p_pe": "???",
            "bachelor_p_probnnp": "???",
            "bachelor_p_track_chi2ndof": "???",
            "bachelor_p_ipchi2": "???",
            
            "kplus_px": "???",
            "kplus_py": "???",
            "kplus_pz": "???",
            "kplus_pe": "???",
            "kplus_probnnk": "???",
            "kplus_track_chi2ndof": "???",
            "kplus_ipchi2": "???",
            
            "kminus_px": "???",
            "kminus_py": "???",
            "kminus_pz": "???",
            "kminus_pe": "???",
            "kminus_probnnk": "???",
            "kminus_track_chi2ndof": "???",
            "kminus_ipchi2": "???",
            
            # Trigger branches
            "l0_global_tis": "Bu_L0GlobalDecision_TIS",
            "l0_phys_tis": "Bu_L0PhysDecision_TIS",
            "l0_hadron_tis": "Bu_L0HadronDecision_TIS",
            "l0_muon_tis": "Bu_L0MuonDecision_TIS",
            "l0_muonhigh_tis": "Bu_L0MuonHighDecision_TIS",
            "l0_dimuon_tis": "Bu_L0DiMuonDecision_TIS",
            "l0_photon_tis": "Bu_L0PhotonDecision_TIS",
            "l0_electron_tis": "Bu_L0ElectronDecision_TIS",
            
            "hlt1_trackmva_tos": "Bu_Hlt1TrackMVADecision_TOS",
            "hlt1_twotrackmva_tos": "Bu_Hlt1TwoTrackMVADecision_TOS",
            
            "hlt2_topo2body_tos": "Bu_Hlt2Topo2BodyDecision_TOS",
            "hlt2_topo3body_tos": "Bu_Hlt2Topo3BodyDecision_TOS",
            "hlt2_topo4body_tos": "Bu_Hlt2Topo4BodyDecision_TOS",
            
            # MC truth (only in MC)
            "true_id": "???_TRUEID",  # Check various particles
            "true_match": "???",
        }
        
        return required
    
    def generate_markdown_documentation(self,
                                       data_branches: Dict,
                                       mc_branches: Dict,
                                       common_unique: Dict,
                                       required: Dict) -> str:
        """
        Generate comprehensive markdown documentation
        
        Saves to: docs/branch_structure.md
        """
        md = "# Branch Structure Documentation\n\n"
        md += "## Summary\n\n"
        md += f"- Total common branches (data + all MC): {len(common_unique['common'])}\n"
        md += f"- Data-only branches: {len(common_unique['data_only'])}\n"
        md += f"- MC-only branches: {len(common_unique['mc_only'])}\n\n"
        
        md += "## Required Branches for Analysis\n\n"
        md += "| Logical Name | Branch Name | Status |\n"
        md += "|--------------|-------------|--------|\n"
        
        for logical, branch in sorted(required.items()):
            if branch.startswith("???"):
                status = "❌ TO BE DISCOVERED"
            elif branch == "DERIVED":
                status = "✓ Computed from other branches"
            elif branch == "NEEDS_CONSTRUCTION":
                status = "✓ Constructed from 4-momenta"
            else:
                status = "✓ Available"
            
            md += f"| `{logical}` | `{branch}` | {status} |\n"
        
        md += "\n## Common Branches (All Files)\n\n"
        md += "Branches present in both data and all MC samples:\n\n"
        for branch in sorted(common_unique['common'])[:50]:  # Show first 50
            md += f"- `{branch}`\n"
        
        if len(common_unique['common']) > 50:
            md += f"\n... and {len(common_unique['common']) - 50} more\n"
        
        md += "\n## Data-Only Branches\n\n"
        for branch in sorted(common_unique['data_only']):
            md += f"- `{branch}`\n"
        
        md += "\n## MC-Only Branches (Truth Information)\n\n"
        for branch in sorted(common_unique['mc_only']):
            md += f"- `{branch}`\n"
        
        md += "\n## Particle Naming Conventions\n\n"
        md += "Need to identify naming pattern for final state particles:\n\n"
        md += "- Lambda (Λ⁰): `L0_*`\n"
        md += "- Bachelor antiproton (p̄): `???` (h0? p? pbar?)\n"
        md += "- K⁺: `???` (h1? Kplus? Kp?)\n"
        md += "- K⁻: `???` (h2? Kminus? Km?)\n"
        md += "- Proton from Lambda decay: `Lp_*` or `L0_p_*`?\n"
        md += "- Pion from Lambda decay: `Lpi_*` or `L0_pi_*`?\n\n"
        
        md += "**ACTION REQUIRED**: Examine a few events and determine naming scheme!\n"
        
        return md
    
    def run_full_inspection(self, output_dir: str = "./docs"):
        """
        Run complete branch inspection and generate documentation
        """
        print("="*60)
        print("PHASE 0: BRANCH STRUCTURE DISCOVERY")
        print("="*60)
        
        print("\n[1/4] Inspecting data files...")
        data_branches = self.inspect_data_files()
        
        print("\n[2/4] Inspecting MC files...")
        mc_branches = self.inspect_mc_files()
        
        print("\n[3/4] Analyzing branch commonality...")
        common_unique = self.find_common_and_unique_branches(data_branches, mc_branches)
        
        print("\n[4/4] Generating documentation...")
        required = self.identify_required_branches()
        markdown = self.generate_markdown_documentation(
            data_branches, mc_branches, common_unique, required
        )
        
        # Save markdown
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        with open(output_path / "branch_structure.md", "w") as f:
            f.write(markdown)
        
        # Save JSON for programmatic access
        branch_map = {
            "data_branches": {k: list(v) for k, v in data_branches.items()},
            "mc_branches": {
                particle: {k: list(v) for k, v in year_data.items()}
                for particle, year_data in mc_branches.items()
            },
            "common": common_unique["common"],
            "data_only": common_unique["data_only"],
            "mc_only": common_unique["mc_only"],
            "required_branches": required
        }
        
        with open(output_path / "branch_structure.json", "w") as f:
            json.dump(branch_map, f, indent=2)
        
        print(f"\n✓ Documentation saved to {output_path}")
        print(f"  - branch_structure.md (human-readable)")
        print(f"  - branch_structure.json (machine-readable)")
        
        # Highlight missing branches
        missing = [k for k, v in required.items() if v.startswith("???")]
        if missing:
            print(f"\n⚠ WARNING: {len(missing)} required branches need discovery:")
            for m in missing:
                print(f"  - {m}")

# Usage
if __name__ == "__main__":
    inspector = BranchInspector(
        data_path="./data",
        mc_path="./mc"
    )
    inspector.run_full_inspection()
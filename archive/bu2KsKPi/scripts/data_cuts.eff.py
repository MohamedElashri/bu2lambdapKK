#!/usr/bin/env python3
"""
Fixed strategy for processing large ROOT files that don't fit in memory by using chunked loading.
With debug information to track the impact of each cut.
"""

import os
import glob
import uproot
import awkward as ak
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Union, Tuple, Optional, Set, Callable

# Define the processing strategy for large data
def process_large_data(
    years: Union[str, List[str]],
    tuple_types: Union[str, List[str]],
    ks_categories: Union[str, List[str]],
    polarities: Union[str, List[str]],
    data_dir: str,
    chunk_size: int = 100000,  # Process in chunks of this many events
    selection_function: Callable = None,  # Function to apply selection cuts
    branches: Optional[List[str]] = None,  # Branches to load
    results_file: str = "selection_results.csv",
    debug_mode: bool = True,  # Enable detailed debugging
    verbose: bool = True
):
    """
    Process large data files in chunks to avoid memory issues.
    
    Args:
        years: Year(s) to process
        tuple_types: Tuple type(s) to process
        ks_categories: K-short categories to process
        polarities: Polarity(ies) to process
        data_dir: Directory containing data files
        chunk_size: Number of events to process at once
        selection_function: Function to apply selection cuts to each chunk
        branches: Specific branches to load (to reduce memory usage)
        results_file: File to save the results
        debug_mode: Whether to enable detailed debugging information
        verbose: Whether to print progress information
    
    Returns:
        Dictionary of results from the processing
    """
    # Normalize input parameters
    if isinstance(years, str) and years != "all":
        years = [years]
    elif years == "all":
        years = ["2015", "2016", "2017", "2018"]
        
    if isinstance(tuple_types, str) and tuple_types != "all":
        tuple_types = [tuple_types]
    elif tuple_types == "all":
        tuple_types = ["KSKmKpPip", "KSKpKpPim"]
        
    if isinstance(ks_categories, str) and ks_categories != "all":
        ks_categories = [ks_categories]
    elif ks_categories == "all":
        ks_categories = ["LL", "DD"]
        
    if isinstance(polarities, str) and polarities != "all":
        polarities = [polarities]
    elif polarities == "all":
        polarities = ["magup", "magdown"]
    
    # Results storage
    results = {
        "total_events": 0,
        "selected_events": 0,
        "selection_efficiency": 0.0,
        "selection_results": [],
        "cut_statistics": {},  # Store statistics about each cut's performance
        "per_category": {}
    }
    
    if verbose:
        print(f"Processing data in chunks of {chunk_size} events")
        print(f"Years: {years}")
        print(f"Tuple types: {tuple_types}")
        print(f"K-short categories: {ks_categories}")
        print(f"Polarities: {polarities}")
    
    # Find all files to process
    files_to_process = []
    
    for year in years:
        for polarity in polarities:
            year_polarity = f"{year}_{polarity}"
            
            for tuple_type in tuple_types:
                file_pattern = os.path.join(data_dir, year_polarity, f"{tuple_type}_reduced.root")
                matching_files = glob.glob(file_pattern)
                
                for file_path in matching_files:
                    for ks_category in ks_categories:
                        tree_name = f"{tuple_type}_{ks_category};1"
                        files_to_process.append({
                            "file_path": file_path,
                            "tree_name": tree_name,
                            "year": year,
                            "polarity": polarity,
                            "tuple_type": tuple_type,
                            "ks_category": ks_category
                        })
    
    if verbose:
        print(f"Found {len(files_to_process)} file/tree combinations to process")
    
    # Process each file in chunks
    for file_info in files_to_process:
        file_path = file_info["file_path"]
        tree_name = file_info["tree_name"]
        category_key = f"{file_info['year']}_{file_info['polarity']}_{file_info['tuple_type']}_{file_info['ks_category']}"
        
        # Initialize category counters if not exists
        if category_key not in results["per_category"]:
            results["per_category"][category_key] = {
                "total": 0,
                "selected": 0,
                "efficiency": 0.0,
                "cut_stats": {}
            }
        
        if verbose:
            print(f"\nProcessing {file_path}, tree: {tree_name}")
        
        try:
            # Check if the tree exists in the file
            with uproot.open(file_path) as file:
                if tree_name not in file:
                    if verbose:
                        print(f"  Tree {tree_name} not found in {file_path}")
                    continue
                
                # Get total number of entries
                num_entries = file[tree_name].num_entries
                
                if verbose:
                    print(f"  Found {num_entries} entries in the tree")
                
                # Process the tree in chunks
                file_total_events = 0
                file_selected_events = 0
                
                # For debugging, check the first few chunks more thoroughly
                debug_chunk_count = 0
                
                for start in range(0, num_entries, chunk_size):
                    end = min(start + chunk_size, num_entries)
                    debug_this_chunk = debug_mode and debug_chunk_count < 3
                    
                    if verbose:
                        print(f"  Processing chunk {start} to {end} ({end - start} events)")
                    
                    # Load this chunk of data
                    chunk = file[tree_name].arrays(
                        expressions=branches,
                        entry_start=start,
                        entry_stop=end,
                        library="ak"
                    )
                    
                    # Debug: check for fields existence
                    if debug_this_chunk:
                        print(f"    Debug: Fields in chunk: {', '.join(chunk.fields)}")
                        missing_fields = set(branches or []) - set(chunk.fields)
                        if missing_fields:
                            print(f"    Warning: Missing fields: {', '.join(missing_fields)}")
                    
                    # Update total events count
                    chunk_size_actual = len(chunk)
                    file_total_events += chunk_size_actual
                    results["total_events"] += chunk_size_actual
                    results["per_category"][category_key]["total"] += chunk_size_actual
                    
                    # Apply selection function if provided
                    if selection_function is not None:
                        # If debugging, use debug version with extra info
                        if debug_this_chunk:
                            selection_result = selection_function(chunk, debug=True)
                        else:
                            selection_result = selection_function(chunk)
                        
                        # Add metadata to the result
                        selection_result.update({
                            "file_path": file_path,
                            "tree_name": tree_name,
                            "year": file_info["year"],
                            "polarity": file_info["polarity"],
                            "tuple_type": file_info["tuple_type"],
                            "ks_category": file_info["ks_category"],
                            "chunk_start": start,
                            "chunk_end": end,
                            "chunk_size": chunk_size_actual
                        })
                        
                        # Update selected events count
                        if "final_count" in selection_result:
                            selected_count = selection_result["final_count"]
                            file_selected_events += selected_count
                            results["selected_events"] += selected_count
                            results["per_category"][category_key]["selected"] += selected_count
                        
                        # Update per-cut statistics
                        if "cut_stats" in selection_result:
                            for cut_name, stats in selection_result["cut_stats"].items():
                                # Initialize if first time seeing this cut
                                if cut_name not in results["cut_statistics"]:
                                    results["cut_statistics"][cut_name] = {
                                        "total_events": 0,
                                        "passed_events": 0,
                                        "efficiency": 0.0,
                                        "independent_passed": 0,  # Count events passing this cut independently
                                        "independent_efficiency": 0.0
                                    }
                                
                                # Update global cut statistics
                                results["cut_statistics"][cut_name]["total_events"] += stats["total"]
                                results["cut_statistics"][cut_name]["passed_events"] += stats["passed"]
                                results["cut_statistics"][cut_name]["independent_passed"] += stats["independent_passed"]
                                
                                # Update per-category cut statistics
                                if "cut_stats" not in results["per_category"][category_key]:
                                    results["per_category"][category_key]["cut_stats"] = {}
                                
                                if cut_name not in results["per_category"][category_key]["cut_stats"]:
                                    results["per_category"][category_key]["cut_stats"][cut_name] = {
                                        "total": 0,
                                        "passed": 0,
                                        "efficiency": 0.0
                                    }
                                
                                results["per_category"][category_key]["cut_stats"][cut_name]["total"] += stats["total"]
                                results["per_category"][category_key]["cut_stats"][cut_name]["passed"] += stats["passed"]
                        
                        # Store the result
                        results["selection_results"].append(selection_result)
                        
                        if verbose:
                            if "efficiency" in selection_result:
                                print(f"    Selection efficiency: {selection_result['efficiency']*100:.2f}%")
                            if "final_count" in selection_result:
                                print(f"    Selected {selection_result['final_count']} events")
                            
                            # Print individual cut statistics for debugging
                            if debug_this_chunk and "cut_stats" in selection_result:
                                print("    Cut-by-cut breakdown:")
                                for cut_name, stats in selection_result["cut_stats"].items():
                                    eff = (stats["passed"] / stats["total"]) * 100 if stats["total"] > 0 else 0
                                    print(f"      {cut_name}: {stats['passed']}/{stats['total']} = {eff:.2f}%")
                    
                    # Debug counter increment
                    debug_chunk_count += 1
                    
                    # Free memory
                    del chunk
                
                # Calculate file-level efficiency
                if file_total_events > 0:
                    file_efficiency = file_selected_events / file_total_events
                    if verbose:
                        print(f"  File efficiency: {file_efficiency*100:.2f}% ({file_selected_events}/{file_total_events})")
        
        except Exception as e:
            if verbose:
                import traceback
                print(f"  Error processing {file_path}: {str(e)}")
                print(traceback.format_exc())
    
    # Calculate overall efficiency and per-category efficiencies
    if results["total_events"] > 0:
        results["selection_efficiency"] = results["selected_events"] / results["total_events"]
    
    # Calculate per-category efficiencies
    for cat_key, cat_data in results["per_category"].items():
        if cat_data["total"] > 0:
            cat_data["efficiency"] = cat_data["selected"] / cat_data["total"]
            
            # Calculate per-cut efficiencies for this category
            if "cut_stats" in cat_data:
                for cut_name, stats in cat_data["cut_stats"].items():
                    if stats["total"] > 0:
                        stats["efficiency"] = stats["passed"] / stats["total"]
    
    # Calculate per-cut efficiencies
    for cut_name, stats in results["cut_statistics"].items():
        if stats["total_events"] > 0:
            stats["efficiency"] = stats["passed_events"] / stats["total_events"]
            stats["independent_efficiency"] = stats["independent_passed"] / stats["total_events"]
    
    if verbose:
        print("\nProcessing complete!")
        print(f"Total events processed: {results['total_events']}")
        print(f"Total events selected: {results['selected_events']}")
        print(f"Overall selection efficiency: {results['selection_efficiency']*100:.2f}%")
        
        # Print per-cut statistics
        print("\nCut-by-cut statistics:")
        for cut_name, stats in sorted(results["cut_statistics"].items(), 
                                      key=lambda x: x[1]["efficiency"]):
            print(f"  {cut_name}:")
            print(f"    Efficiency: {stats['efficiency']*100:.2f}% ({stats['passed_events']}/{stats['total_events']})")
            print(f"    Independent efficiency: {stats['independent_efficiency']*100:.2f}%")
        
        # Print per-category stats
        print("\nBreakdown by category:")
        for cat_key, cat_data in results["per_category"].items():
            if cat_data["total"] > 0:  # Only show categories with data
                print(f"  {cat_key}:")
                print(f"    Events: {cat_data['total']}")
                print(f"    Selected: {cat_data['selected']}")
                print(f"    Efficiency: {cat_data['efficiency']*100:.2f}%")
    
    # Save more detailed results to CSV
    if results["selection_results"]:
        # Main results dataframe
        results_df = pd.DataFrame(results["selection_results"])
        results_df.to_csv(results_file, index=False)
        
        # Save cut statistics
        cuts_file = results_file.replace('.csv', '_cuts.csv')
        cuts_data = []
        
        for cut_name, stats in results["cut_statistics"].items():
            cuts_data.append({
                "cut_name": cut_name,
                "total_events": stats["total_events"],
                "passed_events": stats["passed_events"],
                "efficiency": stats["efficiency"],
                "independent_efficiency": stats["independent_efficiency"]
            })
        
        cuts_df = pd.DataFrame(cuts_data)
        cuts_df.to_csv(cuts_file, index=False)
        
        # Save category summary
        categories_file = results_file.replace('.csv', '_categories.csv')
        categories_data = []
        
        for cat_key, cat_data in results["per_category"].items():
            if cat_data["total"] > 0:
                categories_data.append({
                    "category": cat_key,
                    "total_events": cat_data["total"],
                    "selected_events": cat_data["selected"],
                    "efficiency": cat_data["efficiency"]
                })
        
        categories_df = pd.DataFrame(categories_data)
        categories_df.to_csv(categories_file, index=False)
        
        if verbose:
            print(f"Detailed results saved to {results_file}")
            print(f"Cut statistics saved to {cuts_file}")
            print(f"Category statistics saved to {categories_file}")
    
    return results

# Example selection function with proper debugging and independent cut statistics
def apply_selection_cuts(data, cut_dict, debug=False):
    """Apply selection cuts to a chunk of data and return results with detailed statistics."""
    initial_count = len(data)
    if initial_count == 0:
        return {
            "initial_count": 0,
            "final_count": 0,
            "efficiency": 0.0,
            "cut_counts": {},
            "cut_efficiencies": {}
        }
    
    # Initialize selection mask and statistics
    cumulative_mask = np.ones(initial_count, dtype=bool)
    cut_counts = {}
    cut_stats = {}
    
    # For debugging purposes, check field statistics
    if debug:
        for field in data.fields:
            try:
                if field in [item[0] for item in cut_dict.values()]:
                    field_data = ak.to_numpy(data[field])
                    print(f"    Field {field} stats:")
                    print(f"      Min: {np.min(field_data)}, Max: {np.max(field_data)}")
                    print(f"      Mean: {np.mean(field_data)}, Median: {np.median(field_data)}")
                    print(f"      NaN count: {np.sum(np.isnan(field_data))}")
                    print(f"      Inf count: {np.sum(np.isinf(field_data))}")
            except Exception as e:
                print(f"    Error analyzing field {field}: {str(e)}")
    
    # First, evaluate each cut independently to see its impact
    independent_cut_masks = {}
    for cut_name, (field, operator, value) in cut_dict.items():
        # Skip if field doesn't exist
        if field not in data.fields:
            if debug:
                print(f"    Warning: Field {field} not found in data")
            continue
        
        try:
            # Create the mask for this cut
            if operator == '>':
                mask = data[field] > value
            elif operator == '<':
                mask = data[field] < value
            elif operator == '>=':
                mask = data[field] >= value
            elif operator == '<=':
                mask = data[field] <= value
            elif operator == '==':
                mask = data[field] == value
            elif operator == '!=':
                mask = data[field] != value
            else:
                if debug:
                    print(f"    Warning: Unsupported operator {operator} for cut {cut_name}")
                continue
            
            # Check for NaN or problematic values before conversion
            if debug:
                try:
                    mask_check = ak.to_numpy(mask)
                    nan_count = np.sum(np.isnan(mask_check))
                    if nan_count > 0:
                        print(f"    Warning: {nan_count} NaN values found in mask for cut {cut_name}")
                except Exception as e:
                    print(f"    Error checking mask for NaNs: {str(e)}")
            
            # Convert to numpy and store independent mask
            mask_np = ak.to_numpy(mask)
            independent_cut_masks[cut_name] = mask_np
            
            # Count events passing this cut independently
            independent_passed = np.sum(mask_np)
            
            # Store statistics for this cut
            cut_stats[cut_name] = {
                "total": initial_count,
                "passed": 0,  # Will be updated with cumulative effect
                "independent_passed": independent_passed,
                "independent_efficiency": independent_passed / initial_count
            }
            
            if debug:
                print(f"    Cut {cut_name} independently passes {independent_passed}/{initial_count} events ({independent_passed/initial_count*100:.2f}%)")
        
        except Exception as e:
            if debug:
                print(f"    Error applying cut {cut_name}: {str(e)}")
            continue
    
    # Now apply cuts sequentially to get cumulative effect
    for cut_name, (field, operator, value) in cut_dict.items():
        # Skip if field doesn't exist or we couldn't create a mask
        if cut_name not in independent_cut_masks:
            continue
        
        # Get the mask from our independent calculations
        mask_np = independent_cut_masks[cut_name]
        
        # Update cumulative mask
        cumulative_mask = cumulative_mask & mask_np
        
        # Store count after this cut
        count_after_this_cut = np.sum(cumulative_mask)
        cut_counts[cut_name] = count_after_this_cut
        
        # Update cumulative statistics for this cut
        cut_stats[cut_name]["passed"] = count_after_this_cut
    
    # Calculate final results
    final_count = np.sum(cumulative_mask)
    efficiency = final_count / initial_count if initial_count > 0 else 0
    
    # Calculate efficiencies for each cut
    cut_efficiencies = {
        cut_name: count / initial_count if initial_count > 0 else 0
        for cut_name, count in cut_counts.items()
    }
    
    # Calculate incremental efficiency (how much each cut reduces the sample)
    incremental_efficiencies = {}
    prev_count = initial_count
    
    for cut_name, count in cut_counts.items():
        incremental_efficiencies[cut_name] = count / prev_count if prev_count > 0 else 0
        prev_count = count
    
    return {
        "initial_count": initial_count,
        "final_count": final_count,
        "efficiency": efficiency,
        "cut_counts": cut_counts,
        "cut_efficiencies": cut_efficiencies,
        "incremental_efficiencies": incremental_efficiencies,
        "cut_stats": cut_stats  # Detailed statistics about each cut
    }

# Example of how to use the chunked processing strategy
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process large ROOT files in chunks.")
    parser.add_argument("--data-dir", required=True, help="Directory containing ROOT files")
    parser.add_argument("--years", default="all", help="Years to process (comma-separated)")
    parser.add_argument("--tuple-types", default="all", help="Tuple types to process (comma-separated)")
    parser.add_argument("--ks-categories", default="all", help="K-short categories to process (comma-separated)")
    parser.add_argument("--polarities", default="all", help="Polarities to process (comma-separated)")
    parser.add_argument("--chunk-size", type=int, default=100000, help="Events to process per chunk")
    parser.add_argument("--output", default="selection_results.csv", help="Output file for results")
    parser.add_argument("--debug", action="store_true", help="Enable detailed debugging")
    parser.add_argument("--plot-cuts", action="store_true", help="Create cut efficiency plots")
    parser.add_argument("--inspect-values", action="store_true", help="Inspect field values distributions")
    
    args = parser.parse_args()
    
    # Parse comma-separated lists
    years = args.years.split(",") if args.years != "all" else "all"
    tuple_types = args.tuple_types.split(",") if args.tuple_types != "all" else "all"
    ks_categories = args.ks_categories.split(",") if args.ks_categories != "all" else "all"
    polarities = args.polarities.split(",") if args.polarities != "all" else "all"
    
    # Define the selection cuts
    loose_cuts = {
        'B_PT_cut': ('B_PT', '>', 2000),
        'B_IPCHI2_cut': ('B_IPCHI2_OWNPV', '<', 9),
        'B_FDCHI2_cut': ('B_FDCHI2_OWNPV', '>', 25),
        'B_ENDVERTEX_CHI2_cut': ('B_ENDVERTEX_CHI2', '<', 20),
        'KS_FDCHI2_cut': ('KS_FDCHI2_OWNPV', '>', 100),
        'KS_MM_cut': ('KS_MM', '>', 470),
        'KS_MM_upper_cut': ('KS_MM', '<', 530),
        'KS_P0_ProbNNpi': ('KS_P0_ProbNNpi', '>', 0.2),
        'P0_ProbNNk_cut': ('P0_ProbNNk', '>', 0.5),
        'P1_ProbNNpi_cut': ('P1_ProbNNpi', '>', 0.2),
        'P2_ProbNNk_cut': ('P2_ProbNNk', '>', 0.2)
    }
    
    # Define minimal set of branches to load (to save memory)
    branches = [
        "B_PT", "B_IPCHI2_OWNPV", "B_FDCHI2_OWNPV", "B_ENDVERTEX_CHI2", "B_MM",
        "KS_FDCHI2_OWNPV", "KS_MM", "KS_P0_ProbNNpi",
        "P0_ProbNNk", "P1_ProbNNpi", "P2_ProbNNk"
    ]
    
    # Create selection function closure with our cuts
    def selection_function(data, debug=False):
        return apply_selection_cuts(data, loose_cuts, debug=debug)
    
    # Option to inspect field values before doing full processing
    if args.inspect_values:
        print("\nInspecting field values in first chunk...")
        
        # Find first available file/tree
        if len(files_to_process) > 0:
            file_info = files_to_process[0]
            file_path = file_info["file_path"]
            tree_name = file_info["tree_name"]
            
            try:
                with uproot.open(file_path) as file:
                    if tree_name in file:
                        # Load a small sample
                        sample_size = min(1000, file[tree_name].num_entries)
                        sample = file[tree_name].arrays(
                            expressions=branches,
                            entry_start=0,
                            entry_stop=sample_size,
                            library="ak"
                        )
                        
                        print(f"Loaded {len(sample)} events for inspection")
                        
                        # Analyze each field used in cuts
                        for cut_name, (field, operator, value) in loose_cuts.items():
                            if field in sample.fields:
                                try:
                                    field_data = ak.to_numpy(sample[field])
                                    print(f"\nField: {field} (used in {cut_name})")
                                    print(f"  Cut: {field} {operator} {value}")
                                    print(f"  Min: {np.min(field_data)}, Max: {np.max(field_data)}")
                                    print(f"  Mean: {np.mean(field_data)}, Median: {np.median(field_data)}")
                                    
                                    # Calculate how many events pass this cut
                                    if operator == '>':
                                        passed = np.sum(field_data > value)
                                    elif operator == '<':
                                        passed = np.sum(field_data < value)
                                    elif operator == '>=':
                                        passed = np.sum(field_data >= value)
                                    elif operator == '<=':
                                        passed = np.sum(field_data <= value)
                                    elif operator == '==':
                                        passed = np.sum(field_data == value)
                                    elif operator == '!=':
                                        passed = np.sum(field_data != value)
                                    else:
                                        passed = 0
                                    
                                    print(f"  Events passing cut: {passed}/{len(field_data)} ({passed/len(field_data)*100:.2f}%)")
                                    
                                    # Quick histogram
                                    bins = min(20, len(np.unique(field_data)))
                                    hist, bin_edges = np.histogram(field_data, bins=bins)
                                    print("  Distribution:")
                                    for i in range(len(hist)):
                                        start, end = bin_edges[i], bin_edges[i+1]
                                        print(f"    [{start:.2f} - {end:.2f}]: {hist[i]} events")
                                except Exception as e:
                                    print(f"  Error analyzing {field}: {str(e)}")
                            else:
                                print(f"\nField {field} not found in sample data")
            except Exception as e:
                print(f"Error during field inspection: {str(e)}")
        
        print("\nEnd of field inspection")
        if not input("Continue with full processing? (y/n): ").lower().startswith('y'):
            print("Processing cancelled")
            exit(0)
    
    # Process the data in chunks
    results = process_large_data(
        years=years,
        tuple_types=tuple_types,
        ks_categories=ks_categories,
        polarities=polarities,
        data_dir=args.data_dir,
        chunk_size=args.chunk_size,
        selection_function=selection_function,
        branches=branches,
        results_file=args.output,
        debug_mode=args.debug,
        verbose=True
    )
    
    # Print summary of results
    print("\nResults Summary:")
    print(f"Total events: {results['total_events']}")
    print(f"Selected events: {results['selected_events']}")
    print(f"Overall efficiency: {results['selection_efficiency']*100:.2f}%")
    
    # Create cut efficiency plots if requested
    if args.plot_cuts and "cut_statistics" in results:
        cut_file = args.output.replace('.csv', '_cuts.csv')
        
        # Plot cut efficiencies
        plt.figure(figsize=(12, 8))
        
        # Prepare data for plotting
        cut_names = []
        efficiencies = []
        independent_efficiencies = []
        
        for cut_name, stats in sorted(results["cut_statistics"].items(), 
                                      key=lambda x: x[1]["efficiency"]):
            cut_names.append(cut_name)
            efficiencies.append(stats["efficiency"] * 100)
            independent_efficiencies.append(stats["independent_efficiency"] * 100)
        
        # Plot
        x = np.arange(len(cut_names))
        width = 0.35
        
        plt.bar(x - width/2, efficiencies, width, label='Cumulative Efficiency')
        plt.bar(x + width/2, independent_efficiencies, width, label='Independent Efficiency')
        
        plt.xlabel('Cut')
        plt.ylabel('Efficiency (%)')
        plt.title('Cut Efficiencies')
        plt.xticks(x, cut_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(args.output.replace('.csv', '_cut_efficiencies.png'))
        print(f"Cut efficiency plot saved to {args.output.replace('.csv', '_cut_efficiencies.png')}")
        
        # Create a waterfall chart showing how each cut reduces the event count
        plt.figure(figsize=(14, 8))
        
        # Extract data for waterfall chart
        cut_order = sorted(results["cut_statistics"].items(),
                          key=lambda x: x[1]["efficiency"], reverse=True)
        
        labels = ['Initial']
        values = [results["total_events"]]
        
        for cut_name, stats in cut_order:
            labels.append(cut_name)
            values.append(stats["passed_events"])
        
        # Calculate the differences (waterfall steps)
        edges = []
        for i in range(len(values)-1):
            edges.append(values[i] - values[i+1])
        
        # Plot steps
        cumulative = values[0]
        y_positions = [values[0]]
        
        for i in range(len(edges)):
            plt.bar(i+0.5, -edges[i], bottom=cumulative, width=0.8, color='red', alpha=0.7)
            cumulative -= edges[i]
            y_positions.append(cumulative)
        
        # Plot initial and final values
        plt.bar(0, values[0], width=0.8, color='blue', alpha=0.7)
        plt.bar(len(edges)+0.5, values[-1], width=0.8, color='green', alpha=0.7)
        
        # Labels
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
        plt.ylabel('Events')
        plt.title('Event Reduction by Cut (Waterfall Chart)')
        
        # Add count labels
        for i, v in enumerate(y_positions):
            plt.text(i, v + (0.02 * values[0]), f'{int(v)}', ha='center')
        
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(args.output.replace('.csv', '_waterfall.pdf'))
        print(f"Waterfall chart saved to {args.output.replace('.csv', '_waterfall.pdf')}")
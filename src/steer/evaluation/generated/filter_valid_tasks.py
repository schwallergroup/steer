#!/usr/bin/env python
"""
Filter benchmark tasks to keep only those with valid score variation.

This script:
1. Loads evaluation results from run_evaluation.py
2. Filters for tasks that:
   - Didn't give an error
   - Have non-zero standard deviation
   - Have non-nan standard deviation
3. Creates a filtered benchmark JSON with only valid tasks
4. Reports statistics on filtering
"""

import json
import sys
import numpy as np
from pathlib import Path


def filter_valid_tasks(
    evaluation_results_file: str,
    original_benchmark_file: str,
    output_benchmark_file: str
):
    """Filter benchmark to keep only tasks with valid score variation."""

    # Load evaluation results
    print("Loading evaluation results...")
    with open(evaluation_results_file, 'r') as f:
        results = json.load(f)

    # Load original benchmark
    print("Loading original benchmark...")
    with open(original_benchmark_file, 'r') as f:
        original_benchmark = json.load(f)

    # Create lookup for original entries
    benchmark_by_eval_type = {}
    for entry in original_benchmark:
        eval_type = entry['eval_type']
        benchmark_by_eval_type[eval_type] = entry

    # Filter valid entries
    print("\n" + "="*80)
    print("FILTERING TASKS")
    print("="*80)

    valid_entries = []
    filtered_out = {
        'error': [],
        'zero_std': [],
        'nan_std': []
    }

    for entry in results['entries']:
        eval_type = entry.get('eval_type', 'unknown')

        # Check if it has an error
        if 'error' in entry:
            filtered_out['error'].append({
                'eval_type': eval_type,
                'prompt': entry['prompt'],
                'error': entry['error']
            })
            continue

        # Check if std is nan
        std_score = entry.get('std_score', np.nan)
        if np.isnan(std_score):
            filtered_out['nan_std'].append({
                'eval_type': eval_type,
                'prompt': entry['prompt']
            })
            continue

        # Check if std is zero
        if std_score == 0:
            filtered_out['zero_std'].append({
                'eval_type': eval_type,
                'prompt': entry['prompt'],
                'mean': entry.get('mean_score', 0)
            })
            continue

        # Valid entry - add to filtered benchmark
        if eval_type in benchmark_by_eval_type:
            original_entry = benchmark_by_eval_type[eval_type]
            # Add statistics to the entry
            original_entry['_stats'] = {
                'mean_score': entry['mean_score'],
                'std_score': entry['std_score'],
                'min_score': entry['min_score'],
                'max_score': entry['max_score'],
                'n_routes': entry['n_routes']
            }
            valid_entries.append(original_entry)

    # Print filtering statistics
    print(f"\nOriginal entries: {len(results['entries'])}")
    print(f"Valid entries: {len(valid_entries)}")
    print(f"\nFiltered out:")
    print(f"  - Errors: {len(filtered_out['error'])}")
    print(f"  - Zero std: {len(filtered_out['zero_std'])}")
    print(f"  - NaN std: {len(filtered_out['nan_std'])}")

    # Show examples of filtered out entries
    if filtered_out['error']:
        print(f"\n  Errors (first 3):")
        for item in filtered_out['error'][:3]:
            print(f"    - {item['eval_type']}: {item['error']}")

    if filtered_out['zero_std']:
        print(f"\n  Zero std (first 3):")
        for item in filtered_out['zero_std'][:3]:
            print(f"    - {item['eval_type']}: {item['prompt'][:50]}... (mean={item['mean']:.1f})")

    if filtered_out['nan_std']:
        print(f"\n  NaN std (first 3):")
        for item in filtered_out['nan_std'][:3]:
            print(f"    - {item['eval_type']}: {item['prompt'][:50]}...")

    # Print statistics about valid entries
    print("\n" + "="*80)
    print("VALID TASK STATISTICS")
    print("="*80)

    if valid_entries:
        stds = [e['_stats']['std_score'] for e in valid_entries]
        means = [e['_stats']['mean_score'] for e in valid_entries]
        n_routes = [e['_stats']['n_routes'] for e in valid_entries]

        print(f"\nNumber of valid tasks: {len(valid_entries)}")
        print(f"\nStandard deviation statistics:")
        print(f"  Mean std: {np.mean(stds):.3f}")
        print(f"  Median std: {np.median(stds):.3f}")
        print(f"  Min std: {np.min(stds):.3f}")
        print(f"  Max std: {np.max(stds):.3f}")

        print(f"\nScore statistics:")
        print(f"  Mean score: {np.mean(means):.3f}")
        print(f"  Median score: {np.median(means):.3f}")
        print(f"  Score range: [{np.min(means):.3f}, {np.max(means):.3f}]")

        print(f"\nRoutes per task:")
        print(f"  Mean: {np.mean(n_routes):.1f}")
        print(f"  Range: [{np.min(n_routes)}, {np.max(n_routes)}]")

        # Distribution by molecule
        molecules = {}
        for entry in valid_entries:
            task_id = entry['id']
            if task_id not in molecules:
                molecules[task_id] = []
            molecules[task_id].append(entry['eval_type'])

        print(f"\nDistribution by molecule:")
        for mol_id in sorted(molecules.keys()):
            print(f"  {mol_id}: {len(molecules[mol_id])} tasks")

        # Show top 5 by std
        print(f"\nTop 5 tasks by standard deviation:")
        sorted_entries = sorted(valid_entries, key=lambda e: e['_stats']['std_score'], reverse=True)
        for i, entry in enumerate(sorted_entries[:5], 1):
            stats = entry['_stats']
            print(f"  {i}. {entry['eval_type']}: std={stats['std_score']:.3f}, mean={stats['mean_score']:.3f}")
            print(f"     {entry['prompt'][:70]}...")

    # Save filtered benchmark
    if valid_entries:
        with open(output_benchmark_file, 'w') as f:
            json.dump(valid_entries, f, indent=2)
        print(f"\n✓ Saved {len(valid_entries)} valid tasks to {output_benchmark_file}")
    else:
        print("\n⚠ No valid tasks found - not creating output file")

    print("="*80)

    return {
        'valid_entries': len(valid_entries),
        'filtered_error': len(filtered_out['error']),
        'filtered_zero_std': len(filtered_out['zero_std']),
        'filtered_nan_std': len(filtered_out['nan_std']),
        'total_original': len(results['entries'])
    }


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Filter benchmark to keep only tasks with valid score variation"
    )
    parser.add_argument(
        "--results",
        default="src/steer/evaluation/generated/evaluation_results.json",
        help="Evaluation results JSON from run_evaluation.py"
    )
    parser.add_argument(
        "--benchmark",
        default="src/steer/evaluation/generated/prompt_specs.json",
        help="Original benchmark JSON"
    )
    parser.add_argument(
        "--output",
        default="src/steer/evaluation/generated/prompt_specs_filtered.json",
        help="Output filtered benchmark JSON"
    )

    args = parser.parse_args()

    # Check files exist
    if not Path(args.results).exists():
        print(f"Error: Results file not found: {args.results}")
        print("Please run run_evaluation.py first")
        sys.exit(1)

    if not Path(args.benchmark).exists():
        print(f"Error: Benchmark file not found: {args.benchmark}")
        sys.exit(1)

    # Run filtering
    filter_valid_tasks(
        evaluation_results_file=args.results,
        original_benchmark_file=args.benchmark,
        output_benchmark_file=args.output
    )


if __name__ == "__main__":
    main()

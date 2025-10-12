#!/usr/bin/env python
"""Display evaluation results in a clear, readable format."""

import json
import sys
from pathlib import Path

def display_results(results_file: str = "src/steer/evaluation/generated/evaluation_results.json"):
    """Display evaluation results."""

    with open(results_file, 'r') as f:
        results = json.load(f)

    print("=" * 80)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nBenchmark: {results['benchmark_file']}")
    print(f"Total entries: {results['n_entries']}")

    # Summary statistics
    summary = results['summary']
    print(f"\n{'Status':<30} {'Value':<20}")
    print("-" * 50)
    print(f"{'Successful evaluations':<30} {summary['successful']}/{summary['total_entries']} ({summary['success_rate']:.1f}%)")
    print(f"{'Failed evaluations':<30} {summary['failed']}/{summary['total_entries']}")
    print(f"{'Total routes evaluated':<30} {summary['total_routes_evaluated']}")

    # Overall correlation
    print("\n" + "=" * 80)
    print("OVERALL CORRELATION (across all 223 routes)")
    print("=" * 80)
    overall = results['overall_correlation']
    print(f"Spearman ρ:  {overall['spearman_rho']:>8.3f}  (p={overall['spearman_p']:.6f})")
    print(f"Pearson r:   {overall['pearson_r']:>8.3f}  (p={overall['pearson_p']:.6f})")
    print(f"MAE:         {overall['mae']:>8.3f}")

    if overall['spearman_p'] < 0.05:
        print("\n✓ Statistically significant correlation (p < 0.05)")
    else:
        print("\n⚠ Not statistically significant (p >= 0.05)")

    # Individual entries
    print("\n" + "=" * 80)
    print("INDIVIDUAL ENTRY RESULTS")
    print("=" * 80)

    for i, entry in enumerate(results['entries'], 1):
        print(f"\n[{i}/{results['n_entries']}] {entry['prompt']}")
        print(f"    SMILES: {entry.get('smiles', 'N/A')}")

        if 'error' in entry:
            print(f"    ✗ ERROR: {entry['error']}")
            continue

        print(f"    ✓ Routes evaluated: {entry['n_routes']}")

        corr = entry.get('correlation', {})
        rho = corr.get('spearman_rho', float('nan'))

        # Check if correlation is valid
        import math
        if math.isnan(rho):
            print(f"    Spearman ρ: nan (all routes got same score)")
            print(f"    └─ Mean GT score: {entry.get('mean_gt_score', 0):.2f}")
            print(f"    └─ Mean LM score: {entry.get('mean_lm_score', 0):.2f}")
        else:
            p_val = corr.get('spearman_p', 1.0)
            sig = "**" if p_val < 0.05 else ""
            print(f"    Spearman ρ: {rho:>7.3f} (p={p_val:.3f}) {sig}")
            print(f"    Pearson r:  {corr.get('pearson_r', 0):>7.3f}")
            print(f"    └─ Mean GT score: {entry.get('mean_gt_score', 0):.2f}")
            print(f"    └─ Mean LM score: {entry.get('mean_lm_score', 0):.2f}")

        print(f"    MAE: {corr.get('mae', 0):.3f}")

    # Interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print("""
Mode 1 Evaluation uses PRE-COMPUTED LLM scores from route generation.
These scores reflect the LLM's general assessment of route quality,
NOT specific evaluation of the features in your benchmark.

Key observations:
- Many 'nan' correlations indicate all routes got the same score
- This is EXPECTED because LLM wasn't asked about specific features
- Negative correlation (-0.197) may indicate feature/score inversion

For FEATURE-SPECIFIC evaluation, use Mode 2:
    python src/steer/evaluation/generated/run_evaluation_with_llm.py

Mode 2 calls the LLM with your specific feature prompts and should
show better correlations as the LLM focuses on each feature.
""")

    print("=" * 80)

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Display evaluation results")
    parser.add_argument(
        "--results",
        default="src/steer/evaluation/generated/evaluation_results.json",
        help="Path to evaluation results JSON"
    )

    args = parser.parse_args()

    if not Path(args.results).exists():
        print(f"Error: Results file not found: {args.results}")
        print("\nRun evaluation first:")
        print("  python src/steer/evaluation/generated/run_evaluation.py")
        sys.exit(1)

    display_results(args.results)

if __name__ == "__main__":
    main()

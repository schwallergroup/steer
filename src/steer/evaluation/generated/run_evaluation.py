"""Run evaluation on generated benchmark.

This script evaluates the generated benchmark by:
1. Loading routes and generated evaluation codes
2. Computing ground truth scores for each route
3. Extracting LLM scores from route data
4. Computing correlation between ground truth and LLM scores
"""

import json
import os
import sys
import importlib.util
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
from scipy.stats import spearmanr, pearsonr
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def load_benchmark(benchmark_file: str) -> List[Dict[str, Any]]:
    """Load benchmark specification."""
    with open(benchmark_file, 'r') as f:
        return json.load(f)


def load_routes(routes_dir: str, source_file: str) -> List[Dict[str, Any]]:
    """Load route data for a target molecule."""
    file_path = os.path.join(routes_dir, source_file)
    with open(file_path, 'r') as f:
        return json.load(f)


def import_evaluation_class(code_file: str, class_name: str):
    """Dynamically import an evaluation class from a Python file."""
    spec = importlib.util.spec_from_file_location(class_name, code_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec from {code_file}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[class_name] = module
    spec.loader.exec_module(module)

    # Try to find the class in the module
    if hasattr(module, class_name):
        return getattr(module, class_name)

    # If not found by exact name, look for any class that looks like an evaluator
    # Import base classes to check inheritance
    try:
        # from steer.evaluation.synthesis.eval_types.base import BaseScoring
        # from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase
        from standalone_base import BaseScoring, MultiRxnCondBase
        base_classes = (BaseScoring, MultiRxnCondBase)
    except ImportError:
        # If we can't import base classes, just look for any class
        base_classes = None

    for attr_name in dir(module):
        if attr_name.startswith('_'):  # Skip private attributes
            continue
        attr = getattr(module, attr_name)
        if isinstance(attr, type):
            # If we have base classes, check inheritance
            if base_classes:
                try:
                    if issubclass(attr, base_classes) and attr not in base_classes:
                        return attr
                except TypeError:
                    continue
            # Otherwise just look for plausible class names
            elif attr_name not in ['Dict', 'Tuple', 'Chem', 'BaseScoring', 'MultiRxnCondBase']:
                return attr

    raise ImportError(f"Could not find evaluation class in {code_file}")


def run_single_evaluation(
    entry: Dict[str, Any],
    routes_dir: str,
    codes_dir: str
) -> Dict[str, Any]:
    """Run evaluation for a single benchmark entry.

    Returns:
        Dictionary with ground truth scores and statistics
    """
    try:
        # Get source file
        source_file = entry.get('_source', {}).get('file', None)
        if not source_file:
            return {'error': 'No source file specified in benchmark entry'}

        # Load routes
        routes = load_routes(routes_dir, source_file)

        # Import evaluation class
        eval_type = entry['eval_type']
        feature_id = int(eval_type.split('_')[1])
        code_file = os.path.join(codes_dir, f"feature_{feature_id:03d}.py")

        if not os.path.exists(code_file):
            return {'error': f'Code file not found: {code_file}'}

        EvalClass = import_evaluation_class(code_file, eval_type)

        # Instantiate evaluator
        eval_config = entry['eval_config']
        evaluator = EvalClass(eval_config)

        # Run evaluation - only use ground truth scores (first return value)
        ground_truth_scores, _ = evaluator(routes)

        return {
            'success': True,
            'prompt': entry['prompt'],
            'smiles': entry['smiles'],
            'eval_type': eval_type,
            'n_routes': len(routes),
            'ground_truth_scores': ground_truth_scores
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'prompt': entry.get('prompt', 'N/A'),
            'eval_type': entry.get('eval_type', 'N/A')
        }


def compute_correlation(
    ground_truth: List[float],
    lm_scores: List[float]
) -> Dict[str, float]:
    """Compute correlation metrics."""

    if len(ground_truth) < 2 or len(lm_scores) < 2:
        return {
            'spearman_rho': np.nan,
            'spearman_p': np.nan,
            'pearson_r': np.nan,
            'pearson_p': np.nan,
            'mae': np.nan
        }

    try:
        spearman_rho, spearman_p = spearmanr(ground_truth, lm_scores)
        pearson_r, pearson_p = pearsonr(ground_truth, lm_scores)
        mae = np.mean(np.abs(np.array(ground_truth) - np.array(lm_scores)))

        return {
            'spearman_rho': float(spearman_rho),
            'spearman_p': float(spearman_p),
            'pearson_r': float(pearson_r),
            'pearson_p': float(pearson_p),
            'mae': float(mae)
        }
    except Exception as e:
        print(f"Warning: Correlation computation failed: {e}")
        return {
            'spearman_rho': np.nan,
            'spearman_p': np.nan,
            'pearson_r': np.nan,
            'pearson_p': np.nan,
            'mae': np.nan
        }


def evaluate_benchmark(
    benchmark_file: str,
    routes_dir: str,
    codes_dir: str,
    output_file: str = None
) -> Dict[str, Any]:
    """Evaluate the entire benchmark.

    Args:
        benchmark_file: Path to benchmark JSON
        routes_dir: Directory with route data
        codes_dir: Directory with evaluation code files
        output_file: Optional path to save results

    Returns:
        Evaluation results dictionary
    """
    print("="*80)
    print("BENCHMARK EVALUATION - Score Variation Analysis")
    print("="*80)

    # Load benchmark
    benchmark = load_benchmark(benchmark_file)
    print(f"\nLoaded benchmark: {len(benchmark)} entries")

    results = {
        'benchmark_file': benchmark_file,
        'n_entries': len(benchmark),
        'entries': [],
        'summary': {}
    }

    successful = 0
    failed = 0

    # Process each benchmark entry
    for idx, entry in enumerate(benchmark):
        print(f"\n[{idx+1}/{len(benchmark)}] {entry['prompt']}")

        eval_result = run_single_evaluation(entry, routes_dir, codes_dir)

        if eval_result.get('success'):
            successful += 1

            # Compute statistics for ground truth scores
            gt_scores = eval_result['ground_truth_scores']
            mean_score = float(np.mean(gt_scores))
            std_score = float(np.std(gt_scores))
            min_score = float(np.min(gt_scores))
            max_score = float(np.max(gt_scores))

            print(f"  ✓ Routes: {eval_result['n_routes']}")
            print(f"    Mean: {mean_score:.3f}, Std: {std_score:.3f}")
            print(f"    Range: [{min_score:.3f}, {max_score:.3f}]")

            results['entries'].append({
                'prompt': entry['prompt'],
                'eval_type': eval_result['eval_type'],
                'smiles': entry['smiles'],
                'n_routes': eval_result['n_routes'],
                'mean_score': mean_score,
                'std_score': std_score,
                'min_score': min_score,
                'max_score': max_score,
                'scores': gt_scores  # Keep raw scores for filtering
            })

        else:
            failed += 1
            print(f"  ✗ Error: {eval_result['error']}")

            results['entries'].append({
                'prompt': entry['prompt'],
                'eval_type': eval_result.get('eval_type', entry.get('eval_type', 'unknown')),
                'error': eval_result['error']
            })

    print("\n" + "="*80)
    print("OVERALL RESULTS")
    print("="*80)
    print(f"Total entries: {len(benchmark)}")
    print(f"Successful: {successful} ({100*successful/len(benchmark):.1f}%)")
    print(f"Failed: {failed} ({100*failed/len(benchmark):.1f}%)")
    print("="*80)

    # Summary statistics
    results['summary'] = {
        'total_entries': len(benchmark),
        'successful': successful,
        'failed': failed,
        'success_rate': successful / len(benchmark) if benchmark else 0
    }

    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Saved results to {output_file}")

    return results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate LLM performance on generated benchmark"
    )
    parser.add_argument(
        "--benchmark",
        default="src/steer/evaluation/generated/generated_prompt_specs_sampled.json",
        help="Benchmark specification JSON file"
    )
    parser.add_argument(
        "--routes-dir",
        default="data/outputs/2025-10-12_093739",
        help="Directory with route data"
    )
    parser.add_argument(
        "--codes-dir",
        default="src/steer/evaluation/generated/eval_types_v2",
        help="Directory with evaluation code files"
    )
    parser.add_argument(
        "--output",
        default="src/steer/evaluation/generated/evaluation_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--use-v2",
        action="store_true",
        help="Use V2 generated files (with filtering)"
    )

    args = parser.parse_args()

    # Adjust paths for V2 if requested
    if args.use_v2:
        if 'generated_prompt_specs.json' in args.benchmark and 'v2' not in args.benchmark:
            args.benchmark = args.benchmark.replace('.json', '_v2.json')
        if 'eval_types' in args.codes_dir and 'v2' not in args.codes_dir:
            args.codes_dir = args.codes_dir.replace('eval_types', 'eval_types_v2')
        if 'evaluation_results.json' in args.output and 'v2' not in args.output:
            args.output = args.output.replace('.json', '_v2.json')

    print(f"Benchmark: {args.benchmark}")
    print(f"Routes: {args.routes_dir}")
    print(f"Codes: {args.codes_dir}")
    print(f"Output: {args.output}")

    # Check files exist
    if not os.path.exists(args.benchmark):
        print(f"Error: Benchmark file not found: {args.benchmark}")
        print("Please generate the benchmark first using pipeline.py or pipeline_v2.py")
        sys.exit(1)

    # Run evaluation
    evaluate_benchmark(
        benchmark_file=args.benchmark,
        routes_dir=args.routes_dir,
        codes_dir=args.codes_dir,
        output_file=args.output
    )


if __name__ == "__main__":
    main()

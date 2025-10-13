"""Run evaluation on generated benchmark with live LLM scoring.

This version actually calls an LLM to score routes, rather than using pre-computed scores.
It integrates with the existing Task-based evaluation system from steer.evaluation.synthesis
"""

import json
import os
import sys
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
from scipy.stats import spearmanr, pearsonr

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import existing steer infrastructure
from steer.evaluation.synthesis.tasks import Task, EVAL_CLASSES
from steer.evaluation.synthesis import load_default_tasks, run_task, mae
from steer.llm.sequential import LM


def register_generated_eval_classes(codes_dir: str, metadata_file: str):
    """Register generated evaluation classes with the Task system.

    Args:
        codes_dir: Directory with generated evaluation code
        metadata_file: JSON file with code metadata
    """
    import importlib.util

    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    registered = []

    for feature_id_str, info in metadata.items():
        if 'error' in info:
            continue

        feature_id = int(feature_id_str)
        class_name = info['class_name']
        code_file = os.path.join(codes_dir, f"feature_{feature_id:03d}.py")

        if not os.path.exists(code_file):
            continue

        try:
            # Dynamically import the evaluation class
            spec = importlib.util.spec_from_file_location(class_name, code_file)
            if spec is None or spec.loader is None:
                continue

            module = importlib.util.module_from_spec(spec)
            sys.modules[class_name] = module
            spec.loader.exec_module(module)

            # Find the evaluation class
            eval_class = None
            for attr_name in dir(module):
                if attr_name.startswith('_'):
                    continue
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and attr_name not in ['Dict', 'Tuple', 'BaseScoring', 'MultiRxnCondBase']:
                    eval_class = attr
                    break

            if eval_class:
                # Register it
                EVAL_CLASSES[class_name] = eval_class
                registered.append(class_name)

        except Exception as e:
            print(f"Warning: Could not register {class_name}: {e}")
            continue

    print(f"✓ Registered {len(registered)} evaluation classes")
    return registered


def load_benchmark_as_tasks(benchmark_file: str) -> List[Task]:
    """Load generated benchmark and convert to Task objects.

    Args:
        benchmark_file: Path to benchmark JSON

    Returns:
        List of Task objects
    """
    with open(benchmark_file, 'r') as f:
        benchmark = json.load(f)

    tasks = []
    for entry in benchmark:
        try:
            # Create Task object
            # Note: Task expects eval_type to be in EVAL_CLASSES
            task = Task(
                id=entry['id'],
                smiles=entry['smiles'],
                prompt=entry['prompt'],
                eval_type=entry['eval_type'],
                eval_config=entry['eval_config']
            )
            tasks.append(task)
        except Exception as e:
            print(f"Warning: Could not create task for {entry['prompt']}: {e}")
            continue

    return tasks


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
        mae_val = np.mean(np.abs(np.array(ground_truth) - np.array(lm_scores)))

        return {
            'spearman_rho': float(spearman_rho),
            'spearman_p': float(spearman_p),
            'pearson_r': float(pearson_r),
            'pearson_p': float(pearson_p),
            'mae': float(mae_val)
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


def evaluate_benchmark_with_llm(
    benchmark_file: str,
    codes_dir: str,
    codes_metadata_file: str,
    cache_path: str,
    output_file: str,
    model: str = "gpt-4o",
    prompt: str = "steer.llm.prompts.route_opt",
    max_routes: int = 200,
    use_wandb: bool = False
) -> Dict[str, Any]:
    """Evaluate benchmark by calling LLM to score routes.

    This is the main evaluation function that:
    1. Registers generated evaluation classes
    2. Loads benchmark as Tasks
    3. For each task:
       - Loads precomputed routes from cache
       - Calls LLM to score each route
       - Computes ground truth using evaluation code
       - Compares LLM scores vs ground truth

    Args:
        benchmark_file: Path to benchmark JSON
        codes_dir: Directory with generated evaluation code
        codes_metadata_file: Metadata for generated codes
        cache_path: Directory with precomputed routes
        output_file: Path to save results
        model: LLM model to use (default: gpt-4o)
        prompt: Prompt template to use
        max_routes: Maximum routes to evaluate per task
        use_wandb: Whether to log to Weights & Biases

    Returns:
        Evaluation results dictionary
    """
    print("="*80)
    print("BENCHMARK EVALUATION WITH LLM")
    print("="*80)

    # Optional: Initialize wandb
    if use_wandb:
        import wandb
        wandb.init(
            project="steer-generated-benchmark",
            config={
                "model": model,
                "prompt": prompt,
                "benchmark": benchmark_file,
                "max_routes": max_routes
            }
        )

    # Register generated evaluation classes
    print("\nRegistering evaluation classes...")
    register_generated_eval_classes(codes_dir, codes_metadata_file)

    # Load benchmark as tasks
    print(f"\nLoading benchmark from {benchmark_file}...")
    tasks = load_benchmark_as_tasks(benchmark_file)
    print(f"Loaded {len(tasks)} tasks")

    # Initialize LLM
    print(f"\nInitializing LLM (model={model})...")
    lm = LM(
        prompt=prompt,
        model=model,
        vision=False,
        project_name="steer-generated-benchmark"
    )

    # Evaluate each task
    results = {
        'benchmark_file': benchmark_file,
        'model': model,
        'prompt': prompt,
        'n_tasks': len(tasks),
        'tasks': [],
        'summary': {
            'total_evaluated': 0,
            'total_failed': 0,
            'overall_mae': 0,
            'overall_correlation': 0
        }
    }

    all_gt_scores = []
    all_lm_scores = []

    for idx, task in enumerate(tasks):
        print(f"\n[{idx+1}/{len(tasks)}] {task.prompt}")
        print(f"  Task ID: {task.id}")

        try:
            # Run task (loads routes from cache, calls LLM to score them)
            routes = run_task(
                lm=lm,
                task=task,
                n=max_routes,
                nclusters=0,
                cache_path=cache_path,
                results_path=os.path.dirname(output_file)
            )

            if routes is None:
                print(f"  ✗ No routes found in cache")
                results['tasks'].append({
                    'task_id': task.id,
                    'prompt': task.prompt,
                    'error': 'No routes in cache'
                })
                results['summary']['total_failed'] += 1
                continue

            # Evaluate: get ground truth and LLM scores
            gt_scores, lm_scores = task.evaluate(routes)

            # Compute metrics
            mae_val = mae(gt_scores, lm_scores)

            if len(gt_scores) >= 2:
                corr_val = np.corrcoef(gt_scores, lm_scores)[0, 1]
            else:
                corr_val = np.nan

            correlation = compute_correlation(gt_scores, lm_scores)

            print(f"  ✓ Routes: {len(routes)}")
            print(f"    Spearman ρ: {correlation['spearman_rho']:.3f} (p={correlation['spearman_p']:.3f})")
            print(f"    MAE: {mae_val:.3f}")

            # Store results
            results['tasks'].append({
                'task_id': task.id,
                'prompt': task.prompt,
                'smiles': task.smiles,
                'n_routes': len(routes),
                'mae': float(mae_val),
                'correlation': float(corr_val) if not np.isnan(corr_val) else None,
                'correlation_details': correlation,
                'mean_gt_score': float(np.mean(gt_scores)),
                'mean_lm_score': float(np.mean(lm_scores))
            })

            # Accumulate for overall stats
            all_gt_scores.extend(gt_scores)
            all_lm_scores.extend(lm_scores)
            results['summary']['total_evaluated'] += 1
            results['summary']['overall_mae'] += mae_val
            if not np.isnan(corr_val):
                results['summary']['overall_correlation'] += corr_val

            # Log to wandb if enabled
            if use_wandb:
                import wandb
                wandb.log({
                    f"mae_{task.id}": mae_val,
                    f"corr_{task.id}": corr_val if not np.isnan(corr_val) else 0
                })

        except Exception as e:
            import traceback
            print(f"  ✗ Error: {e}")
            print(f"  Traceback: {traceback.format_exc()}")

            results['tasks'].append({
                'task_id': task.id,
                'prompt': task.prompt,
                'error': str(e)
            })
            results['summary']['total_failed'] += 1

    # Compute overall statistics
    n_evaluated = results['summary']['total_evaluated']
    if n_evaluated > 0:
        results['summary']['mean_mae'] = results['summary']['overall_mae'] / n_evaluated
        results['summary']['mean_correlation'] = results['summary']['overall_correlation'] / n_evaluated

        # Overall correlation across all routes
        if len(all_gt_scores) >= 2:
            overall_corr = compute_correlation(all_gt_scores, all_lm_scores)
            results['summary']['overall_correlation_all_routes'] = overall_corr

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total tasks: {len(tasks)}")
    print(f"Successfully evaluated: {n_evaluated}")
    print(f"Failed: {results['summary']['total_failed']}")
    if n_evaluated > 0:
        print(f"\nMean MAE: {results['summary']['mean_mae']:.3f}")
        print(f"Mean correlation: {results['summary']['mean_correlation']:.3f}")
        if 'overall_correlation_all_routes' in results['summary']:
            overall = results['summary']['overall_correlation_all_routes']
            print(f"\nOverall (all routes):")
            print(f"  Spearman ρ: {overall['spearman_rho']:.3f} (p={overall['spearman_p']:.6f})")
            print(f"  Pearson r: {overall['pearson_r']:.3f}")
            print(f"  MAE: {overall['mae']:.3f}")
    print("="*80)

    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved results to {output_file}")

    # Finish wandb
    if use_wandb:
        import wandb
        if n_evaluated > 0:
            wandb.log({
                "mean_mae": results['summary']['mean_mae'],
                "mean_corr": results['summary']['mean_correlation']
            })
        wandb.finish()

    return results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate benchmark with live LLM scoring"
    )
    parser.add_argument(
        "--benchmark",
        default="src/steer/evaluation/generated/generated_prompt_specs_sampled.json",
        help="Benchmark specification JSON"
    )
    parser.add_argument(
        "--codes-dir",
        default="src/steer/evaluation/generated/eval_types_v2",
        help="Directory with evaluation code"
    )
    parser.add_argument(
        "--codes-metadata",
        default="src/steer/evaluation/generated/eval_types/generated_codes_metadata.json",
        help="Metadata for generated codes"
    )
    parser.add_argument(
        "--cache-path",
        default="data/outputs/2025-10-12_093739",
        help="Directory with precomputed routes"
    )
    parser.add_argument(
        "--output",
        default="src/steer/evaluation/generated/evaluation_results_with_llm.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="LLM model to use"
    )
    parser.add_argument(
        "--prompt",
        default="steer.llm.prompts.route_opt",
        help="Prompt template"
    )
    parser.add_argument(
        "--max-routes",
        type=int,
        default=200,
        help="Maximum routes per task"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log to Weights & Biases"
    )
    parser.add_argument(
        "--use-v2",
        action="store_true",
        help="Use V2 generated files"
    )

    args = parser.parse_args()

    # Adjust paths for V2 if requested
    if args.use_v2:
        if 'generated_prompt_specs.json' in args.benchmark and 'v2' not in args.benchmark:
            args.benchmark = args.benchmark.replace('.json', '_v2.json')
        if 'eval_types' in args.codes_dir and 'v2' not in args.codes_dir:
            args.codes_dir = args.codes_dir.replace('eval_types', 'eval_types_v2')
            args.codes_metadata = args.codes_metadata.replace('eval_types', 'eval_types_v2')
        if 'evaluation_results' in args.output and 'v2' not in args.output:
            args.output = args.output.replace('.json', '_v2.json')

    # Check files exist
    if not os.path.exists(args.benchmark):
        print(f"Error: Benchmark file not found: {args.benchmark}")
        sys.exit(1)

    if not os.path.exists(args.codes_metadata):
        print(f"Error: Codes metadata not found: {args.codes_metadata}")
        sys.exit(1)

    if not os.path.exists(args.cache_path):
        print(f"Error: Cache path not found: {args.cache_path}")
        print("This should be the directory with precomputed route JSON files")
        sys.exit(1)

    # Run evaluation
    evaluate_benchmark_with_llm(
        benchmark_file=args.benchmark,
        codes_dir=args.codes_dir,
        codes_metadata_file=args.codes_metadata,
        cache_path=args.cache_path,
        output_file=args.output,
        model=args.model,
        prompt=args.prompt,
        max_routes=args.max_routes,
        use_wandb=args.wandb
    )


if __name__ == "__main__":
    main()

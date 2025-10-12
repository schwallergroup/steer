#!/usr/bin/env python
"""
Run generated benchmark using the existing steer CLI interface.

This script registers generated evaluation classes and then uses the same
interface as the standard steer benchmark, ensuring compatibility and
code reuse.

Usage:
    python src/steer/evaluation/generated/run_with_cli.py --model gpt-4o
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

def main():
    """Run the generated benchmark using the existing CLI interface."""
    import argparse
    import json
    from datetime import datetime

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Run generated benchmark using steer CLI interface"
    )
    parser.add_argument(
        "--benchmark",
        default="src/steer/evaluation/generated/",
        help="Path to generated benchmark JSON"
    )
    parser.add_argument(
        "--cache-path",
        default="data/outputs/2025-10-12_093739",
        help="Path to cached routes"
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="Model to use for LLM evaluation"
    )
    parser.add_argument(
        "--max-routes",
        type=int,
        default=200,
        help="Maximum number of routes to evaluate per task"
    )
    parser.add_argument(
        "--vision",
        action="store_true",
        help="Use vision mode for reactions"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: auto-generated timestamp)"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging"
    )

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.benchmark).exists():
        print(f"Error: Benchmark file not found: {args.benchmark}")
        sys.exit(1)

    if not Path(args.cache_path).exists():
        print(f"Error: Cache path not found: {args.cache_path}")
        sys.exit(1)

    # Register generated classes
    print("=" * 80)
    print("REGISTERING GENERATED EVALUATION CLASSES")
    print("=" * 80)

    from steer.evaluation.generated import register_with_steer
    registered_classes = register_with_steer()

    print(f"\n✓ Ready to evaluate {len(registered_classes)} generated features\n")

    # Set up output directory
    if args.output_dir is None:
        dt_name = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        args.output_dir = f"data/outputs/generated_benchmark_{dt_name}"
    os.makedirs(args.output_dir, exist_ok=True)

    # Load tasks using existing infrastructure
    print("=" * 80)
    print("LOADING BENCHMARK TASKS")
    print("=" * 80)

    from steer.evaluation.synthesis import load_default_tasks
    tasks = load_default_tasks(args.benchmark)

    print(f"\n✓ Loaded {len(tasks)} tasks from benchmark\n")

    # Initialize LLM
    print("=" * 80)
    print("INITIALIZING LLM")
    print("=" * 80)

    from steer.llm.sequential import LM

    prompt = "steer.llm.prompts.route_opt"
    project = "steer-generated-benchmark"

    lm = LM(
        prompt=prompt,
        model=args.model,
        vision=args.vision,
        project_name=project,
    )

    print(f"\n✓ Initialized {args.model} with prompt: {prompt}\n")

    # Initialize wandb if enabled
    if not args.no_wandb:
        print("=" * 80)
        print("INITIALIZING WANDB")
        print("=" * 80)
        import wandb
        wandb.init(
            project=project,
            config={
                "model": args.model,
                "vision": args.vision,
                "prompt": prompt,
                "benchmark": args.benchmark,
                "cache_path": args.cache_path,
                "max_routes": args.max_routes,
                "output_dir": args.output_dir,
            },
        )
        print("\n✓ Wandb initialized\n")
    else:
        wandb = None

    # Run evaluation using existing infrastructure
    print("=" * 80)
    print("RUNNING EVALUATION")
    print("=" * 80)
    print(f"\nBenchmark: {args.benchmark}")
    print(f"Cache: {args.cache_path}")
    print(f"Model: {args.model}")
    print(f"Max routes per task: {args.max_routes}")
    print(f"Output: {args.output_dir}\n")

    from steer.evaluation.synthesis import run_task, mae
    import numpy as np

    results = []
    metrics = {
        "MAE": 0,
        "Corr": 0,
        "n_valid": 0,
    }

    for i, task in enumerate(tasks, 1):
        print(f"\n[{i}/{len(tasks)}] {task.prompt}")
        print(f"  SMILES: {task.smiles}")
        print(f"  Eval type: {task.eval_type}")

        try:
            # Use existing run_task infrastructure
            routes = run_task(
                lm,
                task,
                n=args.max_routes,
                nclusters=0,
                cache_path=args.cache_path,
                results_path=args.output_dir,
            )

            if routes is None:
                print(f"  ✗ No routes found")
                results.append({
                    "task_id": task.id,
                    "prompt": task.prompt,
                    "error": "No routes found in cache"
                })
                continue

            # Evaluate using task.evaluate() - THIS IS THE KEY!
            # This calls the LLM and computes ground truth
            gt_scores, lm_scores = task.evaluate(routes)

            print(f"  ✓ Evaluated {len(routes)} routes")

            # Compute metrics
            mae_val = mae(gt_scores, lm_scores)
            cor_val = np.corrcoef(gt_scores, lm_scores)[0, 1]

            print(f"    MAE: {mae_val:.3f}")
            print(f"    Corr: {cor_val:.3f}")

            # Store results
            results.append({
                "task_id": task.id,
                "prompt": task.prompt,
                "smiles": task.smiles,
                "eval_type": task.eval_type,
                "n_routes": len(routes),
                "mae": float(mae_val),
                "correlation": float(cor_val),
                "mean_gt_score": float(np.mean(gt_scores)),
                "mean_lm_score": float(np.mean(lm_scores)),
            })

            metrics["MAE"] += mae_val
            metrics["Corr"] += cor_val
            metrics["n_valid"] += 1

            # Log to wandb if enabled
            if wandb:
                wandb.log({
                    f"mae_{task.id}": mae_val,
                    f"corr_{task.id}": cor_val
                })

        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()

            results.append({
                "task_id": task.id,
                "prompt": task.prompt,
                "error": str(e)
            })

    # Compute overall metrics
    print("\n" + "=" * 80)
    print("OVERALL RESULTS")
    print("=" * 80)

    if metrics["n_valid"] > 0:
        mean_mae = metrics["MAE"] / metrics["n_valid"]
        mean_corr = metrics["Corr"] / metrics["n_valid"]

        print(f"\nSuccessful evaluations: {metrics['n_valid']}/{len(tasks)}")
        print(f"Mean MAE: {mean_mae:.3f}")
        print(f"Mean Correlation: {mean_corr:.3f}")

        if wandb:
            wandb.log({
                "mean_mae": mean_mae,
                "mean_corr": mean_corr,
                "n_successful": metrics["n_valid"],
                "n_total": len(tasks),
            })
    else:
        print("\n✗ No successful evaluations")

    # Save results
    results_file = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            "benchmark": args.benchmark,
            "model": args.model,
            "cache_path": args.cache_path,
            "max_routes": args.max_routes,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tasks": len(tasks),
                "successful": metrics["n_valid"],
                "failed": len(tasks) - metrics["n_valid"],
                "mean_mae": metrics["MAE"] / metrics["n_valid"] if metrics["n_valid"] > 0 else None,
                "mean_correlation": metrics["Corr"] / metrics["n_valid"] if metrics["n_valid"] > 0 else None,
            },
            "results": results,
        }, f, indent=2)

    print(f"\n✓ Results saved to: {results_file}")
    print("=" * 80)

    if wandb:
        wandb.finish()


if __name__ == "__main__":
    main()

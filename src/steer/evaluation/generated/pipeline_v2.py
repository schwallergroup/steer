"""Master pipeline for generating USPTO190 benchmark (v2 - improved).

Improvements over v1:
- Quality filtering during feature extraction
- Parallel API calls for 3-5x speedup
- Better statistics and reporting
"""

import os
import sys
import argparse
from pathlib import Path

# Import our v2 modules
from extract_features_v2 import extract_features_from_all_routes
from code_generator_v2 import generate_all_evaluation_codes
from create_benchmark import (
    create_benchmark_spec,
    create_sampling_strategy,
    analyze_benchmark_coverage
)


def run_pipeline_v2(
    input_dir: str,
    output_dir: str,
    api_key: str,
    max_files: int = None,
    max_workers: int = 5,
    routes_per_file: int = None,
    samples_per_molecule: int = 3
):
    """Run the complete benchmark generation pipeline (v2 - improved).

    Args:
        input_dir: Directory with route JSON files
        output_dir: Base directory for outputs
        api_key: Anthropic API key
        max_files: Maximum number of files to process (None for all)
        max_workers: Number of parallel API calls (default: 5)
        routes_per_file: Maximum routes per file (None for all)
        samples_per_molecule: Number of features per molecule in sampled benchmark
    """
    print("="*80)
    print("USPTO190 BENCHMARK GENERATION PIPELINE V2")
    print("(With quality filtering and parallelization)")
    print("="*80)

    # Setup paths
    features_file = os.path.join(output_dir, "extracted_features_v2.json")
    eval_codes_dir = os.path.join(output_dir, "eval_types_v2")
    codes_metadata_file = os.path.join(eval_codes_dir, "generated_codes_metadata.json")
    benchmark_file = os.path.join(output_dir, "generated_prompt_specs_v2.json")

    # Step 1: Extract features (with quality filtering)
    print("\n" + "="*80)
    print("STEP 1: EXTRACTING FEATURES (with quality filtering)")
    print("="*80)

    if os.path.exists(features_file):
        print(f"⚠ Features file already exists: {features_file}")
        response = input("Do you want to re-extract features? (y/n): ")
        if response.lower() != 'y':
            print("Skipping feature extraction...")
        else:
            extract_features_from_all_routes(
                input_dir=input_dir,
                output_file=features_file,
                api_key=api_key,
                max_files=max_files,
                max_workers=max_workers,
                routes_per_file=routes_per_file
            )
    else:
        extract_features_from_all_routes(
            input_dir=input_dir,
            output_file=features_file,
            api_key=api_key,
            max_files=max_files,
            max_workers=max_workers,
            routes_per_file=routes_per_file
        )

    # Step 2: Generate evaluation codes (parallelized)
    print("\n" + "="*80)
    print("STEP 2: GENERATING EVALUATION CODE (parallelized)")
    print("="*80)

    if os.path.exists(codes_metadata_file):
        print(f"⚠ Evaluation codes already exist: {eval_codes_dir}")
        response = input("Do you want to re-generate codes? (y/n): ")
        if response.lower() != 'y':
            print("Skipping code generation...")
        else:
            generate_all_evaluation_codes(
                features_file=features_file,
                output_dir=eval_codes_dir,
                api_key=api_key,
                max_workers=max_workers
            )
    else:
        generate_all_evaluation_codes(
            features_file=features_file,
            output_dir=eval_codes_dir,
            api_key=api_key,
            max_workers=max_workers
        )

    # Step 3: Create benchmark specification
    print("\n" + "="*80)
    print("STEP 3: CREATING BENCHMARK SPECIFICATION")
    print("="*80)

    benchmark_specs = create_benchmark_spec(
        features_file=features_file,
        codes_metadata_file=codes_metadata_file,
        output_file=benchmark_file
    )

    # Analyze coverage
    analyze_benchmark_coverage(benchmark_specs)

    # Create sampled version
    print("\n" + "="*80)
    print("STEP 4: CREATING SAMPLED BENCHMARK")
    print("="*80)

    create_sampling_strategy(
        benchmark_specs=benchmark_specs,
        output_file=benchmark_file,
        n_samples_per_molecule=samples_per_molecule
    )

    # Final summary
    print("\n" + "="*80)
    print("PIPELINE V2 COMPLETE!")
    print("="*80)
    print(f"✓ Features extracted (with filtering): {features_file}")
    print(f"✓ Evaluation codes generated: {eval_codes_dir}")
    print(f"✓ Full benchmark: {benchmark_file}")
    print(f"✓ Sampled benchmark: {benchmark_file.replace('.json', '_sampled.json')}")
    print("\nNext steps:")
    print("  1. Review: Check the statistics in *_stats.json files")
    print("  2. Test: Try the sampled benchmark in your evaluation framework")
    print("  3. Expand: Run on all 96 molecules if satisfied")
    print("="*80)


def main():
    """Main entry point with CLI."""
    parser = argparse.ArgumentParser(
        description="Generate USPTO190 benchmark (v2 - improved)"
    )
    parser.add_argument(
        "--input-dir",
        default="data/outputs/2025-10-12_093739",
        help="Directory containing route JSON files"
    )
    parser.add_argument(
        "--output-dir",
        default="src/steer/evaluation/generated",
        help="Output directory for generated files"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to process (default: all)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Number of parallel API calls (default: 5, max: 10)"
    )
    parser.add_argument(
        "--routes-per-file",
        type=int,
        default=None,
        help="Maximum routes per file (default: all)"
    )
    parser.add_argument(
        "--samples-per-molecule",
        type=int,
        default=3,
        help="Number of features per molecule in sampled benchmark (default: 3)"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Test mode: 3 files, 5 routes each, 3 workers"
    )

    args = parser.parse_args()

    # Get API key
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    # Override in test mode
    if args.test_mode:
        args.max_files = 3
        args.routes_per_file = 5
        args.max_workers = 3
        print("⚠️  TEST MODE: 3 files, 5 routes each, 3 workers")

    # Validate max_workers
    if args.max_workers > 10:
        print("⚠️  Warning: max_workers > 10 may hit API rate limits")
        print("   Reducing to 10")
        args.max_workers = 10

    # Run pipeline
    run_pipeline_v2(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        api_key=api_key,
        max_files=args.max_files,
        max_workers=args.max_workers,
        routes_per_file=args.routes_per_file,
        samples_per_molecule=args.samples_per_molecule
    )


if __name__ == "__main__":
    main()

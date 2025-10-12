"""Create final benchmark specification from generated features and evaluation codes.

This script combines the extracted features, generated evaluation codes,
and route data to produce the final prompt_specs.json benchmark file.
"""

import json
import os
import hashlib
from typing import Dict, List, Any


def generate_id(smiles: str, prompt: str) -> str:
    """Generate a unique ID for a benchmark case."""
    content = f"{smiles}:{prompt}"
    return hashlib.md5(content.encode()).hexdigest()


def create_benchmark_spec(
    features_file: str,
    codes_metadata_file: str,
    output_file: str,
    config_defaults: Dict[str, Any] = None
) -> List[Dict[str, Any]]:
    """Create benchmark specification JSON.

    Args:
        features_file: JSON file with extracted features
        codes_metadata_file: JSON file with generated code metadata
        output_file: Path to save benchmark specification
        config_defaults: Default configuration for route search

    Returns:
        List of benchmark specifications
    """
    # Load inputs
    with open(features_file, 'r') as f:
        all_features = json.load(f)

    with open(codes_metadata_file, 'r') as f:
        codes_metadata = json.load(f)

    # Default config for route search
    if config_defaults is None:
        config_defaults = {
            "iter_lim": 500,
            "time_lim": 1000,
            "max_tree": 20
        }

    # Build benchmark specs
    benchmark_specs = []

    for feature_id_str, metadata in codes_metadata.items():
        # Skip entries with errors
        if 'error' in metadata:
            continue

        feature_id = int(feature_id_str)
        class_name = metadata['class_name']

        # Create benchmark entry
        # Note: Generated code expects config["parameters"], so wrap parameters
        # Use source file name (without .json) as id for CLI compatibility
        task_id = metadata.get('source_file', '').replace('.json', '') if metadata.get('source_file') else generate_id(metadata['smiles'], metadata['prompt'])

        spec = {
            "smiles": metadata['smiles'],
            "prompt": metadata['prompt'],
            "id": task_id,
            "eval_type": class_name,
            "eval_config": {
                "parameters": metadata['parameters']
            },
            "config": config_defaults.copy()
        }

        # Add optional fields
        if metadata.get('rationale'):
            spec['_rationale'] = metadata['rationale']

        if metadata.get('source_file'):
            spec['_source'] = {
                'file': metadata['source_file'],
                'route_index': metadata['route_index']
            }

        benchmark_specs.append(spec)

    # Save benchmark specification
    with open(output_file, 'w') as f:
        json.dump(benchmark_specs, f, indent=2)

    print(f"\n✓ Created benchmark with {len(benchmark_specs)} entries")
    print(f"✓ Saved to {output_file}")

    return benchmark_specs


def create_sampling_strategy(
    benchmark_specs: List[Dict[str, Any]],
    output_file: str,
    n_samples_per_molecule: int = 3
) -> Dict[str, Any]:
    """Create a sampling strategy to ensure diverse benchmark coverage.

    For each molecule, select up to n_samples_per_molecule features
    that are most diverse and interesting.

    Args:
        benchmark_specs: List of benchmark specifications
        output_file: Path to save sampling strategy
        n_samples_per_molecule: Number of features per molecule to include

    Returns:
        Sampling strategy dictionary
    """
    # Group by SMILES
    by_molecule = {}
    for spec in benchmark_specs:
        smiles = spec['smiles']
        if smiles not in by_molecule:
            by_molecule[smiles] = []
        by_molecule[smiles].append(spec)

    # Sample from each molecule
    sampled = []
    sampling_info = {
        'total_molecules': len(by_molecule),
        'total_features': len(benchmark_specs),
        'sampled_features': 0,
        'sampling_strategy': 'diverse_per_molecule',
        'n_per_molecule': n_samples_per_molecule
    }

    for smiles, specs in by_molecule.items():
        # For now, simple strategy: take first n_samples_per_molecule
        # In a more sophisticated version, we could:
        # - Prioritize different feature types
        # - Avoid similar prompts
        # - Select based on rationale quality
        selected = specs[:n_samples_per_molecule]
        sampled.extend(selected)

    sampling_info['sampled_features'] = len(sampled)

    # Save sampled benchmark
    sampled_file = output_file.replace('.json', '_sampled.json')
    with open(sampled_file, 'w') as f:
        json.dump(sampled, f, indent=2)

    # Save sampling info
    info_file = output_file.replace('.json', '_sampling_info.json')
    with open(info_file, 'w') as f:
        json.dump(sampling_info, f, indent=2)

    print(f"\n✓ Sampled {len(sampled)} features from {len(by_molecule)} molecules")
    print(f"✓ Saved sampled benchmark to {sampled_file}")
    print(f"✓ Saved sampling info to {info_file}")

    return sampling_info


def analyze_benchmark_coverage(
    benchmark_specs: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Analyze the coverage of the benchmark.

    Returns statistics about feature types, molecules, prompts, etc.
    """
    # Count by feature type
    feature_types = {}
    for spec in benchmark_specs:
        ft = spec.get('eval_type', 'unknown')
        feature_types[ft] = feature_types.get(ft, 0) + 1

    # Count unique molecules
    unique_smiles = set(spec['smiles'] for spec in benchmark_specs)

    # Prompt length distribution
    prompt_lengths = [len(spec['prompt'].split()) for spec in benchmark_specs]

    analysis = {
        'total_entries': len(benchmark_specs),
        'unique_molecules': len(unique_smiles),
        'feature_type_distribution': feature_types,
        'prompt_length_stats': {
            'min': min(prompt_lengths) if prompt_lengths else 0,
            'max': max(prompt_lengths) if prompt_lengths else 0,
            'avg': sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0
        }
    }

    print("\n" + "="*60)
    print("BENCHMARK COVERAGE ANALYSIS")
    print("="*60)
    print(f"Total entries: {analysis['total_entries']}")
    print(f"Unique molecules: {analysis['unique_molecules']}")
    print(f"Avg features per molecule: {analysis['total_entries'] / analysis['unique_molecules']:.1f}")
    print(f"\nFeature type distribution:")
    for ft, count in sorted(feature_types.items(), key=lambda x: -x[1]):
        print(f"  {ft}: {count}")
    print(f"\nPrompt length (words):")
    print(f"  Min: {analysis['prompt_length_stats']['min']}")
    print(f"  Max: {analysis['prompt_length_stats']['max']}")
    print(f"  Avg: {analysis['prompt_length_stats']['avg']:.1f}")
    print("="*60)

    return analysis


def main():
    """Main entry point."""
    # Paths
    features_file = "src/steer/evaluation/generated/extracted_features.json"
    codes_metadata_file = "src/steer/evaluation/generated/eval_types/generated_codes_metadata.json"
    output_file = "src/steer/evaluation/generated/generated_prompt_specs.json"

    # Create benchmark specification
    benchmark_specs = create_benchmark_spec(
        features_file=features_file,
        codes_metadata_file=codes_metadata_file,
        output_file=output_file
    )

    # Analyze coverage
    analyze_benchmark_coverage(benchmark_specs)

    # Create sampled version
    create_sampling_strategy(
        benchmark_specs=benchmark_specs,
        output_file=output_file,
        n_samples_per_molecule=3
    )


if __name__ == "__main__":
    main()

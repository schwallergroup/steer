"""Generate evaluation code from extracted features (v2 - parallelized).

Improvements over v1:
- Parallel code generation for faster processing
- Better error handling
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import anthropic


CODE_GENERATION_PROMPT = """You are an expert Python programmer specializing in cheminformatics. Your task is to generate evaluation code for a synthesis route feature.

Here are examples of existing evaluation classes:

<example_1>
# RingBreakDepth - checks at what depth a ring-breaking/forming reaction occurs
class RingBreakDepth(BaseScoring):
    def __init__(self, config: Dict):
        self.condition_type = config["target_depth"]["type"]
        self.target_depth = config["target_depth"]["value"]

    def route_scoring(self, x) -> float:
        if self.condition_type == "bool":
            if self.target_depth == -1:  # Positive if condition not met
                return 1 if x < 0 else 0
        else:
            if x < 0:
                return 0
            return abs(x - self.target_depth)

    def hit_condition(self, d):
        return d.get("metadata", {{}}).get("policy_name") == "ringbreaker"
</example_1>

<example_2>
# SpecificBondBreak - checks if a specific bond is broken
class SpecificBondBreak(BaseScoring):
    def __init__(self, config):
        self.atom_1 = config["bond_to_break"]["atom_1"]
        self.atom_2 = config["bond_to_break"]["atom_2"]

    def route_scoring(self, x):
        if x < 0:
            return 0  # Disconnection doesn't happen
        else:
            return 1 - x  # Late-stage disconnection is better

    def hit_condition(self, d):
        rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
        prod = Chem.MolFromSmiles(rxn[0])
        reacts = [Chem.MolFromSmiles(r) for r in rxn[1].split(".")]

        if (self.atom_1 in [a.GetAtomMapNum() for a in prod.GetAtoms()]) and \\
           (self.atom_2 in [a.GetAtomMapNum() for a in prod.GetAtoms()]):
            for r in reacts:
                if (self.atom_1 in [a.GetAtomMapNum() for a in r.GetAtoms()]) ^ \\
                   (self.atom_2 in [a.GetAtomMapNum() for a in r.GetAtoms()]):
                    return True
        return False
</example_2>

<example_3>
# MultiRxnCond - checks presence/absence of multiple reaction types
class MultiRxnCond(MultiRxnCondBase):
    def __init__(self, config):
        self.allow_piperidine = config.get("allow_piperidine") or False
        # ... other ring flags

    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        pip = any(self.detect_piperidine(r) for r in reactions)
        # ... check other conditions

        condition = pip == self.allow_piperidine  # and other checks
        return condition, len(reactions)

    def detect_piperidine(self, rxn):
        pattern = "C1CN[CH2]CC1"
        return self.detect_specific_break(rxn, pattern)
</example_3>

Key concepts:
1. **BaseScoring classes** traverse the route tree using `condition_depth()` which does BFS to find when `hit_condition()` is met
2. **hit_condition()** checks a single reaction node
3. **route_scoring()** converts the depth fraction to a 0-10 score
4. **MultiRxnCondBase classes** check all reactions in the tree, not just depth
5. Use RDKit's `Chem.MolFromSmarts()` and `HasSubstructMatch()` for substructure detection

Now generate code for this feature:

Feature:
{feature}

Generate a complete Python class that:
1. Inherits from the appropriate base class (BaseScoring or MultiRxnCondBase)
2. Implements the necessary methods
3. Uses RDKit for chemistry operations
4. Follows the patterns shown in the examples

Return ONLY the Python class code, with a clear class name and docstring. Do not include imports or surrounding text.
"""


def generate_evaluation_class(
    feature: Dict[str, Any],
    client: anthropic.Anthropic
) -> str:
    """Generate evaluation class code for a feature."""

    feature_json = json.dumps(feature, indent=2)
    prompt = CODE_GENERATION_PROMPT.format(feature=feature_json)

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    code = message.content[0].text

    # Clean up code blocks if present
    if "```python" in code:
        code_start = code.find("```python") + 9
        code_end = code.find("```", code_start)
        code = code[code_start:code_end].strip()
    elif "```" in code:
        code_start = code.find("```") + 3
        code_end = code.find("```", code_start)
        code = code[code_start:code_end].strip()

    return code


def generate_single_code(
    feature_id: int,
    feature: Dict[str, Any],
    source_file: str,
    route_idx: int,
    smiles: str,
    client: anthropic.Anthropic
) -> tuple[int, Dict[str, Any]]:
    """Generate code for a single feature (for parallel execution).

    Returns:
        (feature_id, result_dict)
    """
    try:
        # Generate evaluation class
        code = generate_evaluation_class(feature, client)

        # Create unique class name
        class_name = f"Feature_{feature_id:03d}"

        return feature_id, {
            'success': True,
            'source_file': source_file,
            'route_index': route_idx,
            'smiles': smiles,
            'prompt': feature['prompt'],
            'feature_type': feature['feature_type'],
            'parameters': feature['parameters'],
            'class_name': class_name,
            'rationale': feature.get('rationale', ''),
            'code': code
        }

    except Exception as e:
        return feature_id, {
            'success': False,
            'error': str(e),
            'feature': feature
        }


def generate_all_evaluation_codes(
    features_file: str,
    output_dir: str,
    api_key: str,
    max_workers: int = 5
) -> Dict[str, Any]:
    """Generate evaluation code files for all extracted features with parallelization.

    Args:
        features_file: JSON file with extracted features
        output_dir: Directory to save generated evaluation classes
        api_key: Anthropic API key
        max_workers: Number of parallel API calls (default: 5)

    Returns:
        Dictionary mapping feature IDs to generated code info
    """
    # Load extracted features
    with open(features_file, 'r') as f:
        all_features = json.load(f)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("="*80)
    print("CODE GENERATION (Parallelized)")
    print("="*80)

    # Collect all features to process
    tasks = []
    feature_id = 0

    for filename, file_features in all_features.items():
        for route_features in file_features:
            route_idx = route_features['route_index']
            smiles = route_features['smiles']

            for feature in route_features['features']:
                feature_id += 1
                tasks.append({
                    'feature_id': feature_id,
                    'feature': feature,
                    'source_file': filename,
                    'route_idx': route_idx,
                    'smiles': smiles
                })

    print(f"\nGenerating code for {len(tasks)} features with {max_workers} parallel workers...")

    generated_codes = {}

    # Generate codes in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create clients
        clients = [anthropic.Anthropic(api_key=api_key) for _ in range(max_workers)]

        # Submit all tasks
        future_to_id = {}
        for idx, task in enumerate(tasks):
            client = clients[idx % max_workers]
            future = executor.submit(
                generate_single_code,
                task['feature_id'],
                task['feature'],
                task['source_file'],
                task['route_idx'],
                task['smiles'],
                client
            )
            future_to_id[future] = task['feature_id']

        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_id):
            feature_id, result = future.result()
            completed += 1

            if result['success']:
                # Save metadata
                generated_codes[feature_id] = {
                    'source_file': result['source_file'],
                    'route_index': result['route_idx'],
                    'smiles': result['smiles'],
                    'prompt': result['prompt'],
                    'feature_type': result['feature_type'],
                    'parameters': result['parameters'],
                    'class_name': result['class_name'],
                    'rationale': result['rationale']
                }

                # Save code file
                code_file = os.path.join(output_dir, f"feature_{feature_id:03d}.py")
                with open(code_file, 'w') as f:
                    f.write(f'"""Generated evaluation code for: {result["prompt"]}"""\n\n')
                    f.write("from typing import Dict, Tuple\n")
                    f.write("from rdkit import Chem\n")
                    f.write("from steer.evaluation.synthesis.eval_types.base import BaseScoring\n")
                    f.write("from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase\n\n")
                    f.write(result['code'])
                    f.write("\n")

                print(f"  [{completed}/{len(tasks)}] Feature {feature_id:03d}: {result['prompt'][:50]} ✓")

            else:
                generated_codes[feature_id] = {
                    'error': result['error'],
                    'feature': result['feature']
                }
                print(f"  [{completed}/{len(tasks)}] Feature {feature_id:03d}: ERROR - {result['error']}")

    # Save metadata
    metadata_file = os.path.join(output_dir, 'generated_codes_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(generated_codes, f, indent=2)

    # Summary
    success_count = sum(1 for v in generated_codes.values() if 'error' not in v)
    print("\n" + "="*80)
    print(f"Generated {len(generated_codes)} evaluation codes")
    print(f"Success: {success_count} ({100*success_count/len(generated_codes):.1f}%)")
    print(f"Failed: {len(generated_codes) - success_count}")
    print(f"✓ Saved metadata to {metadata_file}")
    print("="*80)

    return generated_codes


def main():
    """Main entry point."""
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate evaluation codes (v2 - parallelized)"
    )
    parser.add_argument(
        "--features-file",
        default="src/steer/evaluation/generated/extracted_features_v2.json",
        help="Extracted features JSON file"
    )
    parser.add_argument(
        "--output-dir",
        default="src/steer/evaluation/generated/eval_types_v2",
        help="Output directory for generated code"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Number of parallel API calls (default: 5)"
    )

    args = parser.parse_args()

    # Get API key from environment
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    # Generate evaluation codes
    generate_all_evaluation_codes(
        features_file=args.features_file,
        output_dir=args.output_dir,
        api_key=api_key,
        max_workers=args.max_workers
    )


if __name__ == "__main__":
    main()

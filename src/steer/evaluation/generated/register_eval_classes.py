#!/usr/bin/env python
"""
Register generated evaluation classes with the steer evaluation system.

This allows generated benchmark entries to work with the existing CLI:
    python -m steer.cli synth --bench_spec src/steer/evaluation/generated/generated_prompt_specs_sampled.json bench
"""

import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, Type

# Add path to allow imports
sys.path.insert(0, str(Path(__file__).parent))


def load_evaluation_class(code_file: Path, class_name: str):
    """Dynamically load an evaluation class from a Python file."""
    spec = importlib.util.spec_from_file_location(class_name, code_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {code_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Import base classes to check inheritance
    from steer.evaluation.synthesis.eval_types.base import BaseScoring
    from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase
    base_classes = (BaseScoring, MultiRxnCondBase)

    # Find the evaluation class by checking for inheritance
    for attr_name in dir(module):
        if attr_name.startswith('_'):
            continue
        attr = getattr(module, attr_name)
        if isinstance(attr, type):
            if issubclass(attr, base_classes) and attr not in base_classes:
                return attr

    raise ValueError(f"Could not find evaluation class in {code_file}")


def register_generated_eval_classes(
    codes_dir: str = "src/steer/evaluation/generated/eval_types_v2",
    metadata_file: str = "src/steer/evaluation/generated/eval_types_v2/generated_codes_metadata.json"
) -> Dict[str, Type]:
    """
    Register all generated evaluation classes.

    Returns:
        Dictionary mapping class names to class objects that can be added to EVAL_CLASSES
    """
    codes_path = Path(codes_dir)
    metadata_path = Path(metadata_file)

    if not codes_path.exists():
        raise FileNotFoundError(f"Codes directory not found: {codes_dir}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    registered_classes = {}
    failed = []

    print(f"Registering evaluation classes from {codes_dir}...")

    for feature_id, meta in metadata.items():
        class_name = meta['class_name']
        code_file = codes_path / f"{class_name.lower()}.py"

        if not code_file.exists():
            failed.append((class_name, f"File not found: {code_file}"))
            continue

        try:
            eval_class = load_evaluation_class(code_file, class_name)
            registered_classes[class_name] = eval_class
        except Exception as e:
            failed.append((class_name, str(e)))

    print(f"✓ Registered {len(registered_classes)} evaluation classes")

    if failed:
        print(f"✗ Failed to register {len(failed)} classes:")
        for class_name, error in failed[:5]:  # Show first 5
            print(f"  - {class_name}: {error}")
        if len(failed) > 5:
            print(f"  ... and {len(failed) - 5} more")

    return registered_classes


def update_tasks_module():
    """
    Update the steer.evaluation.synthesis.tasks module to include generated classes.

    This modifies EVAL_CLASSES in-place to include all generated evaluation classes.
    """
    # Import the tasks module
    from steer.evaluation.synthesis.tasks import EVAL_CLASSES

    # Register generated classes
    generated_classes = register_generated_eval_classes()

    # Add to EVAL_CLASSES
    original_count = len(EVAL_CLASSES)
    EVAL_CLASSES.update(generated_classes)

    print(f"\n✓ Updated EVAL_CLASSES: {original_count} → {len(EVAL_CLASSES)} classes")
    print(f"  Added {len(generated_classes)} generated classes")

    return EVAL_CLASSES


def main():
    """Main entry point for testing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Register generated evaluation classes"
    )
    parser.add_argument(
        "--codes-dir",
        default="src/steer/evaluation/generated/eval_types_v2",
        help="Directory containing generated evaluation code"
    )
    parser.add_argument(
        "--metadata",
        default="src/steer/evaluation/generated/eval_types/generated_codes_metadata.json",
        help="Metadata file with class information"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test registration without modifying tasks module"
    )

    args = parser.parse_args()

    if args.test:
        # Just register and show what would be added
        classes = register_generated_eval_classes(args.codes_dir, args.metadata)
        print(f"\nWould register {len(classes)} classes:")
        for name in sorted(classes.keys())[:10]:
            print(f"  - {name}")
        if len(classes) > 10:
            print(f"  ... and {len(classes) - 10} more")
    else:
        # Actually update the tasks module
        update_tasks_module()


if __name__ == "__main__":
    main()

"""Extract synthesis features from LLM route descriptions (v2 - improved).

Improvements over v1:
- Quality filtering during extraction
- Parallel API calls for faster processing
- Better error handling and retry logic
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import anthropic


FEATURE_EXTRACTION_PROMPT = """You are an expert organic chemist analyzing synthetic routes. Your task is to extract 1-3 SHORT, generalizable synthesis features from this detailed route analysis.

IMPORTANT QUALITY CRITERIA - Only extract features that are:
1. **Specific and measurable** - we need to write code to verify them
2. **Strategic and interesting** - focus on key decisions like:
   - Ring formation/breaking timing (early vs late stage)
   - Specific bond disconnections
   - Protecting group strategies (but NOT "redundant" or "unnecessary" ones)
   - Convergent vs linear approaches
3. **Concise** - 5-12 words maximum
4. **Objective** - NOT subjective judgments like "redundant", "unnecessary", "excessive", "inefficient"

GOOD EXAMPLES:
- "Late imidazole ring formation"
- "Early stage thiazole assembly"
- "Convergent synthesis via two fragments"
- "Minimal protecting group strategy"
- "Late stage amide coupling"

BAD EXAMPLES (DO NOT EXTRACT):
- "Redundant protecting group cycling" (subjective)
- "Unnecessary deprotection steps" (subjective)
- "Inefficient route design" (too vague)
- "Complex multi-step approach" (not specific)

Here is the detailed route analysis:

<route_analysis>
{route_analysis}
</route_analysis>

For each feature you extract, provide:
1. A short prompt (5-12 words, objective, no subjective quality judgments)
2. A feature type (one of: ring_break_timing, specific_bond_break, ring_formation_count, reaction_type_presence, protecting_group_strategy, convergent_strategy)
3. The specific parameters needed to evaluate this feature programmatically
4. A brief rationale (1 sentence)

Return your response as a JSON list of features:
```json
[
  {{
    "prompt": "Late thiazole ring formation",
    "feature_type": "ring_break_timing",
    "parameters": {{
      "ring_smarts": "c1scnc1",
      "timing": "late",
      "direction": "formation"
    }},
    "rationale": "Thiazole is formed in the final step via Hantzsch synthesis"
  }}
]
```

Extract 1-3 features maximum. If the route doesn't have any interesting strategic features, return an empty list [].
Focus on features that distinguish this route from alternatives.
"""


QUALITY_FILTER_KEYWORDS = [
    'redundant', 'unnecessary', 'excessive', 'inefficient',
    'poor', 'suboptimal', 'problematic', 'awkward'
]


def load_route_data(file_path: str) -> List[Dict[str, Any]]:
    """Load route data from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def is_high_quality_feature(feature: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """Check if a feature meets quality criteria.

    Returns:
        (is_valid, rejection_reason)
    """
    prompt = feature.get('prompt', '').lower()

    # Check for subjective quality keywords
    for keyword in QUALITY_FILTER_KEYWORDS:
        if keyword in prompt:
            return False, f"contains subjective keyword '{keyword}'"

    # Check prompt length
    words = prompt.split()
    if len(words) < 3:
        return False, "prompt too short (<3 words)"

    if len(words) > 12:
        return False, "prompt too long (>12 words)"

    # Check for required fields
    if not feature.get('feature_type'):
        return False, "missing feature_type"

    if not feature.get('parameters'):
        return False, "missing parameters"

    # Check feature type is valid
    valid_types = {
        'ring_break_timing', 'specific_bond_break', 'ring_formation_count',
        'reaction_type_presence', 'protecting_group_strategy', 'convergent_strategy'
    }
    if feature.get('feature_type') not in valid_types:
        return False, f"invalid feature_type: {feature.get('feature_type')}"

    return True, None


def extract_features_from_route(
    route_data: Dict[str, Any],
    client: anthropic.Anthropic,
    max_retries: int = 2
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Extract synthesis features from a single route using Claude API.

    Returns:
        (features, stats) where stats contains filtering info
    """
    stats = {
        'extracted': 0,
        'filtered': 0,
        'kept': 0,
        'filtered_reasons': []
    }

    # Get the LLM analysis from the route
    lm_response = route_data.get('lmdata', {}).get('response', '')

    if not lm_response:
        return [], stats

    # Call Claude API to extract features with retries
    for attempt in range(max_retries):
        try:
            prompt = FEATURE_EXTRACTION_PROMPT.format(route_analysis=lm_response)

            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            response_text = message.content[0].text

            # Extract JSON from response
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_str = response_text[json_start:json_end].strip()
            else:
                json_start = response_text.find('[')
                json_end = response_text.rfind(']') + 1
                if json_start == -1 or json_end == 0:
                    return [], stats
                json_str = response_text[json_start:json_end].strip()

            try:
                features = json.loads(json_str)
                stats['extracted'] = len(features)

                # Filter features for quality
                filtered_features = []
                for feature in features:
                    is_valid, reason = is_high_quality_feature(feature)
                    if is_valid:
                        filtered_features.append(feature)
                        stats['kept'] += 1
                    else:
                        stats['filtered'] += 1
                        stats['filtered_reasons'].append({
                            'prompt': feature.get('prompt', 'N/A'),
                            'reason': reason
                        })

                return filtered_features, stats

            except json.JSONDecodeError as e:
                if attempt < max_retries - 1:
                    time.sleep(1)  # Wait before retry
                    continue
                print(f"    Failed to parse JSON after {max_retries} attempts: {e}")
                return [], stats

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            print(f"    Error extracting features: {e}")
            return [], stats

    return [], stats


def process_single_route(
    route_idx: int,
    route: Dict[str, Any],
    client: anthropic.Anthropic
) -> tuple[int, Dict[str, Any], Dict[str, Any]]:
    """Process a single route (for parallel execution).

    Returns:
        (route_idx, route_features_dict, stats)
    """
    features, stats = extract_features_from_route(route, client)

    if features:
        return route_idx, {
            'route_index': route_idx,
            'smiles': route.get('smiles', ''),
            'features': features
        }, stats
    else:
        return route_idx, None, stats


def extract_features_from_all_routes(
    input_dir: str,
    output_file: str,
    api_key: str,
    max_files: int = None,
    max_workers: int = 5,
    routes_per_file: int = None
) -> Dict[str, Any]:
    """Extract features from all route files in directory with parallelization.

    Args:
        input_dir: Directory containing route JSON files
        output_file: Path to save extracted features
        api_key: Anthropic API key
        max_files: Maximum number of files to process (None for all)
        max_workers: Number of parallel API calls (default: 5)
        routes_per_file: Maximum routes to process per file (None for all)

    Returns:
        Dictionary mapping file names to extracted features
    """
    # Get all JSON files
    route_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.json')])

    if max_files:
        route_files = route_files[:max_files]

    all_features = {}
    total_stats = {
        'total_routes': 0,
        'total_extracted': 0,
        'total_filtered': 0,
        'total_kept': 0,
        'filtered_reasons': []
    }

    print(f"Processing {len(route_files)} files with up to {max_workers} parallel workers...")

    for file_idx, filename in enumerate(route_files):
        print(f"\n[{file_idx+1}/{len(route_files)}] Processing {filename}...")

        file_path = os.path.join(input_dir, filename)
        routes = load_route_data(file_path)

        if routes_per_file:
            # Sample a random subset of routes
            import random
            routes = random.sample(routes, min(routes_per_file, len(routes)))

        print(f"  {len(routes)} routes to process...")

        # Process routes in parallel
        file_features = []
        file_stats = {
            'extracted': 0,
            'filtered': 0,
            'kept': 0
        }

        # Create a client for each worker
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create clients for each task
            clients = [anthropic.Anthropic(api_key=api_key) for _ in range(max_workers)]

            # Submit all tasks
            future_to_route = {}
            for route_idx, route in enumerate(routes):
                # Round-robin assign clients
                client = clients[route_idx % max_workers]
                future = executor.submit(
                    process_single_route,
                    route_idx,
                    route,
                    client
                )
                future_to_route[future] = route_idx

            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_route):
                route_idx, result, stats = future.result()
                completed += 1

                if result:
                    file_features.append(result)
                    print(f"    [{completed}/{len(routes)}] Route {route_idx}: "
                          f"{stats['kept']} features kept "
                          f"({stats['filtered']} filtered)")
                else:
                    print(f"    [{completed}/{len(routes)}] Route {route_idx}: "
                          f"no features")

                # Update stats
                file_stats['extracted'] += stats['extracted']
                file_stats['filtered'] += stats['filtered']
                file_stats['kept'] += stats['kept']
                total_stats['filtered_reasons'].extend(stats['filtered_reasons'])

        # Store results
        all_features[filename] = file_features
        total_stats['total_routes'] += len(routes)
        total_stats['total_extracted'] += file_stats['extracted']
        total_stats['total_filtered'] += file_stats['filtered']
        total_stats['total_kept'] += file_stats['kept']

        print(f"  File summary: {file_stats['kept']} features kept, "
              f"{file_stats['filtered']} filtered")

    # Save results
    with open(output_file, 'w') as f:
        json.dump(all_features, f, indent=2)

    # Save stats
    stats_file = output_file.replace('.json', '_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(total_stats, f, indent=2)

    # Print summary
    print("\n" + "="*80)
    print("EXTRACTION SUMMARY")
    print("="*80)
    print(f"Total routes processed: {total_stats['total_routes']}")
    print(f"Features extracted: {total_stats['total_extracted']}")
    print(f"Features filtered out: {total_stats['total_filtered']} "
          f"({100*total_stats['total_filtered']/max(total_stats['total_extracted'],1):.1f}%)")
    print(f"Features kept: {total_stats['total_kept']} "
          f"({100*total_stats['total_kept']/max(total_stats['total_extracted'],1):.1f}%)")

    # Top filter reasons
    if total_stats['filtered_reasons']:
        print(f"\nTop rejection reasons:")
        from collections import Counter
        reasons = [r['reason'] for r in total_stats['filtered_reasons']]
        for reason, count in Counter(reasons).most_common(5):
            print(f"  {count:3d}x: {reason}")

    print(f"\n✓ Saved extracted features to {output_file}")
    print(f"✓ Saved statistics to {stats_file}")
    print("="*80)

    return all_features


def main():
    """Main entry point."""
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract features from route descriptions (v2 - with filtering and parallelization)"
    )
    parser.add_argument(
        "--input-dir",
        default="data/outputs/2025-10-12_093739",
        help="Directory containing route JSON files"
    )
    parser.add_argument(
        "--output",
        default="src/steer/evaluation/generated/extracted_features_v2.json",
        help="Output file for extracted features"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to process"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Number of parallel API calls (default: 5)"
    )
    parser.add_argument(
        "--routes-per-file",
        type=int,
        default=None,
        help="Maximum routes per file to process"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: 3 files, 5 routes each"
    )

    args = parser.parse_args()

    # Get API key from environment
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    # Test mode
    if args.test:
        args.max_files = 3
        args.routes_per_file = 5
        print("⚠️  TEST MODE: Processing 3 files, 5 routes each")

    # Extract features
    extract_features_from_all_routes(
        input_dir=args.input_dir,
        output_file=args.output,
        api_key=api_key,
        max_files=args.max_files,
        max_workers=args.max_workers,
        routes_per_file=args.routes_per_file
    )


if __name__ == "__main__":
    main()

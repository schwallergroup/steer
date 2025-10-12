# Quick Start Guide: Generating USPTO190 Benchmark

This guide will help you generate a benchmark from USPTO190 route descriptions in just a few steps.

## Prerequisites

1. **API Key**: Set your Anthropic API key
```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

2. **Dependencies**: Ensure required packages are installed
```bash
pip install anthropic rdkit scipy matplotlib seaborn
```

3. **Data**: Route descriptions should be in `data/outputs/2025-10-12_093739/`

## Step-by-Step Walkthrough

### Test Run (Recommended First)

Start with a test run on 3 files to verify everything works:

```bash
cd /home/andres/Documents/steer
python src/steer/evaluation/generated/pipeline.py --test-mode
```

This will:
- Extract features from 3 route files
- Generate evaluation code for each feature
- Create benchmark specifications
- Take ~5-10 minutes

**Expected Output:**
```
================================================================================
USPTO190 BENCHMARK GENERATION PIPELINE
================================================================================

================================================================================
STEP 1: EXTRACTING FEATURES FROM ROUTE DESCRIPTIONS
================================================================================
Processing target_001.json (1/3)...
  Route 1/24... Extracted 2 features
  Route 2/24... Extracted 3 features
  ...

✓ Saved extracted features to src/steer/evaluation/generated/extracted_features.json

================================================================================
STEP 2: GENERATING EVALUATION CODE FOR EACH FEATURE
================================================================================
Processing target_001.json...
  Generating code for feature 1: Late thiazole ring formation
    ✓ Saved to eval_types/feature_001.py
  ...

✓ Generated 15 evaluation codes
✓ Saved metadata to eval_types/generated_codes_metadata.json

================================================================================
STEP 3: CREATING BENCHMARK SPECIFICATION
================================================================================

✓ Created benchmark with 15 entries
✓ Saved to generated_prompt_specs.json

...
```

### Full Pipeline

Once the test run succeeds, run on all files:

```bash
python src/steer/evaluation/generated/pipeline.py
```

This will take ~30-60 minutes depending on:
- Number of route files (96 in USPTO190)
- Number of routes per file (typically 10-30)
- API rate limits

### Custom Configuration

For more control:

```bash
python src/steer/evaluation/generated/pipeline.py \
    --input-dir data/outputs/2025-10-12_093739 \
    --output-dir src/steer/evaluation/generated \
    --max-files 20 \
    --samples-per-molecule 3
```

## Outputs

After successful completion, you'll have:

```
src/steer/evaluation/generated/
├── extracted_features.json                    # All extracted features
├── generated_prompt_specs.json                # Full benchmark
├── generated_prompt_specs_sampled.json        # Sampled benchmark (recommended)
├── generated_prompt_specs_sampling_info.json  # Sampling statistics
└── eval_types/
    ├── feature_001.py                         # Generated evaluation code
    ├── feature_002.py
    ├── ...
    └── generated_codes_metadata.json          # Metadata for all features
```

## Validation

Validate the generated evaluation codes:

```bash
python src/steer/evaluation/generated/validate_codes.py
```

This tests that each generated evaluation class:
- Can be imported
- Can process route data
- Produces valid scores (0-10)

**Expected Output:**
```
================================================================================
VALIDATION REPORT
================================================================================

[001] Late thiazole ring formation
  File: eval_types/feature_001.py
  Testing against: target_001.json
  ✓ Passed all checks
  Scores: min=0.00, max=8.50, mean=4.23

...

================================================================================
SUMMARY
================================================================================
Total codes tested: 45
Passed: 43 (95.6%)
Failed: 2 (4.4%)
================================================================================
```

## Analysis

Analyze the characteristics of extracted features:

```bash
python src/steer/evaluation/generated/analyze_features.py
```

This shows:
- Feature type distribution
- Prompt statistics
- Coverage metrics
- Parameter patterns

## Using the Benchmark

### Option 1: Direct Integration

The generated benchmark is compatible with existing evaluation code:

```python
from steer.evaluation.synthesis.evaluation import evaluate_benchmark

results = evaluate_benchmark(
    benchmark_file="src/steer/evaluation/generated/generated_prompt_specs_sampled.json",
    routes_dir="data/outputs/2025-10-12_093739"
)
```

### Option 2: Custom Evaluation

```python
import json
from eval_types.feature_001 import Feature_001

# Load benchmark entry
with open("generated_prompt_specs_sampled.json") as f:
    benchmark = json.load(f)

entry = benchmark[0]

# Load routes
with open(f"data/outputs/2025-10-12_093739/{entry['_source']['file']}") as f:
    routes = json.load(f)

# Run evaluation
evaluator = Feature_001(config=entry['eval_config'])
ground_truth_scores, lm_scores = evaluator(routes)

# Compute correlation
from scipy.stats import spearmanr
correlation, p_value = spearmanr(ground_truth_scores, lm_scores)
print(f"Correlation: {correlation:.3f} (p={p_value:.3f})")
```

## Troubleshooting

### Issue: API Rate Limits

**Solution**: Add rate limiting in `extract_features.py` or `code_generator.py`:

```python
import time
time.sleep(1)  # Add after each API call
```

### Issue: Import Errors for Generated Code

**Problem**: `ModuleNotFoundError` when importing evaluation classes

**Solution**: Ensure the code uses correct imports:

```python
from steer.evaluation.synthesis.eval_types.base import BaseScoring
```

If issues persist, check the generated code manually in `eval_types/feature_XXX.py`

### Issue: Low Validation Pass Rate

**Problem**: Many generated codes fail validation

**Solutions**:
1. Review failed codes in validation report
2. Check for common patterns in failures
3. Improve code generation prompt in `code_generator.py`
4. Manually fix problematic generated codes

### Issue: Empty Features Extracted

**Problem**: No features extracted from route descriptions

**Solutions**:
1. Check that route files have `lmdata.response` field
2. Verify API key is valid
3. Review feature extraction prompt in `extract_features.py`
4. Check for API errors in console output

## Next Steps

1. **Manual Review**: Review a sample of generated prompts and evaluation codes for quality

2. **Refinement**: Based on validation results, you may want to:
   - Manually fix failing evaluation codes
   - Adjust feature extraction prompts
   - Filter out low-quality features

3. **Expansion**: Generate more diverse features by:
   - Adjusting sampling strategy in `create_benchmark.py`
   - Modifying feature extraction to prioritize specific types
   - Adding new feature type categories

4. **Evaluation**: Run the benchmark on your LLM evaluation pipeline

5. **Iteration**: Based on results, refine the pipeline:
   - Improve feature extraction prompts
   - Add new evaluation code templates
   - Adjust sampling strategy

## Key Files Reference

| File | Purpose |
|------|---------|
| `pipeline.py` | Main orchestration script - run this |
| `extract_features.py` | Extract features from LLM route descriptions |
| `code_generator.py` | Generate evaluation Python code |
| `create_benchmark.py` | Create benchmark JSON specification |
| `validate_codes.py` | Test generated evaluation codes |
| `analyze_features.py` | Analyze feature characteristics |
| `example_usage.py` | Example of using the benchmark |

## Getting Help

1. Check [README.md](README.md) for detailed documentation
2. Review generated files for errors:
   - `extracted_features.json` - feature extraction results
   - `generated_codes_metadata.json` - code generation metadata
   - `validation_report.json` - validation results
3. Examine console output for API errors or exceptions
4. Check individual generated files in `eval_types/` for code issues

## Success Metrics

A successful benchmark generation should have:

- ✅ **Coverage**: At least 1-3 features per molecule
- ✅ **Diversity**: Mix of feature types (ring breaking, bond disconnection, etc.)
- ✅ **Quality**: >90% validation pass rate
- ✅ **Specificity**: Prompts are concise (<15 words) but specific
- ✅ **Reproducibility**: Evaluation codes produce consistent scores

You can check these with:
```bash
python src/steer/evaluation/generated/analyze_features.py
python src/steer/evaluation/generated/validate_codes.py
```

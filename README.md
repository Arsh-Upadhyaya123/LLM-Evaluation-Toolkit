# LLM Response Evaluation Toolkit

### Make LLM output comparisons reproducible—not just anecdotal.

A lightweight command-line evaluator that scores **pre-generated model responses** against reference answers, then exports both aggregate results and per-example diagnostics. It is useful as an initial regression-checking layer when experimenting with prompts, model versions, or answer-generation pipelines.

**Why this project?** LLM outputs are easy to demo and hard to compare consistently. This toolkit turns a collection of responses into structured, repeatable measurements while keeping each metric's limitations explicit.

**Tech:** Python · JSONL · Exact Match · Token F1 · CLI · unittest

## What it does

- **Read JSONL evaluations:** One labeled question, reference, generated answer, and context per line.
- **Score text overlap:** Normalized exact-match and token-level F1.
- **Inspect supplied context:** Report whether answer text is literally contained in the context.
- **Report two levels of detail:** Dataset averages plus case-level scores.
- **Stay model-agnostic:** Works with answer files from any generation system, with no API key required.
- **Validate input:** Detect missing fields, invalid JSON, and duplicate IDs.

## Run it in 30 seconds

Requires **Python 3.10+**. No packages to install.

```bash
python evaluate.py sample_predictions.jsonl --out evaluation_report.json
```

The CLI prints summary metrics; `evaluation_report.json` includes aggregate scores and detailed per-example results.

## Input format

Create a `.jsonl` file with one JSON object per line:

```json
{"id":"q1","question":"What is the capital of France?","reference_answer":"Paris","model_answer":"Paris","context":"Paris is the capital of France."}
```

Required string fields: `id`, `question`, `reference_answer`, `model_answer`, and `context`.

## Understand the metrics

| Metric | What it actually measures |
| --- | --- |
| **Exact-match rate** | Fraction of answers matching the reference after basic normalization |
| **Mean token F1** | Average precision/recall of word overlap with the reference |
| **Answer-contained-in-context rate** | Fraction of answers appearing as literal normalized substrings in supplied context |
| **Per-example diagnostics** | Exact match, token F1, and answer/reference containment flags |

**Important:** Context containment is **not hallucination detection or factual verification**. Lexical metrics can also penalize valid paraphrases and reward some incorrect answers.

## Test

```bash
python -m unittest -v
```

## Project scope and next steps

This tool **does not call an LLM** or judge semantic correctness. The bundled sample responses are synthetic. Extend it with human-labeled evaluation sets, semantic metrics, confidence intervals, per-category breakdowns, and regression thresholds for CI.

**GitHub description:** Model-agnostic LLM answer evaluation CLI with exact match, token F1, JSONL reports and context-overlap diagnostics.

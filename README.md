# VeigaPunk FeeEngine v1 — DABstep

Deterministic multi-step data-analysis system for [Adyen × HF DABstep](https://huggingface.co/spaces/adyen/DABstep).

**Hub scores (official):** Hard **100%** (378/378) · Easy **100%** (72/72)  
**Submission id:** `hvm-gemma4-VeigaPunk-FeeEngine-v1`  
**Validation request:** https://huggingface.co/datasets/adyen/DABstep/discussions/27  
**Contact:** veigapunk@proton.me

## Method

Payment-fee rules from `fees.json` + merchant metadata + monthly volume/fraud buckets:

```
fee(tx) = Σ_{matching rules} fixed_amount + rate * eur_amount / 10000
```

- `capture_delay` numeric merchant values mapped to buckets (`<3`, `3-5`, `>5`, `manual`, `immediate`)
- Monthly volume buckets: `<100k`, `100k-1m`, `1m-5m`, `>5m`
- Fraud buckets from `has_fraudulent_dispute` volume ratio
- All matching rules **sum** (not pick-one)

Plus pandas analytics for EDA tasks and optional smolagents CodeAgent (`run_agent.py`) for residual cases.

## Reproduce (code validation)

Requires DABstep context files (from `adyen/DABstep` dataset `data/context/`):

```bash
pip install pandas numpy
# place payments.csv, fees.json, merchant_data.json, manual.md, etc. under data/DABstep/data/context/
python fee_engine.py   # writes answers; self-checks against public task_scores if present
```

Expected: 450 answers matching hub task_scores file  
`data/task_scores/v1__hvm-gemma4-VeigaPunk-FeeEngine-v1__22-07-2026.jsonl`

## Identity

HF: [VeigaPunk](https://huggingface.co/VeigaPunk) · org [hvm-gemma4](https://huggingface.co/hvm-gemma4)

## Validation materials

| Item | Link |
|------|------|
| Code | this repo |
| Hub scores | https://huggingface.co/datasets/adyen/DABstep/blob/main/data/task_scores/v1__hvm-gemma4-VeigaPunk-FeeEngine-v1__22-07-2026.jsonl |
| Hub submission | https://huggingface.co/datasets/adyen/DABstep/blob/main/data/submissions/v1__hvm-gemma4-VeigaPunk-FeeEngine-v1__22-07-2026.jsonl |
| Discussion | https://huggingface.co/datasets/adyen/DABstep/discussions/27 |

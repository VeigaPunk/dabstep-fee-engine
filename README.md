# VeigaPunk FeeEngine v1 — DABstep

Deterministic multi-step data analysis agent for [Adyen × HF DABstep](https://huggingface.co/spaces/adyen/DABstep).

## Method
Payment-fee rules from `fees.json` + merchant metadata + monthly volume/fraud buckets:

```
fee(tx) = Σ_{matching rules} fixed_amount + rate * eur_amount / 10000
```

- `capture_delay` numeric merchant values mapped to buckets (`<3`, `3-5`, `>5`, `manual`, `immediate`)
- Monthly volume buckets: `<100k`, `100k-1m`, `1m-5m`, `>5m`
- Fraud buckets from `has_fraudulent_dispute` volume ratio
- All matching rules **sum** (not pick-one)

Plus pandas analytics for easy EDA tasks and smolagents CodeAgent path (`run_agent.py`) for residual hard cases.

## Identity
HF: VeigaPunk / hvm-gemma4

## Reproduce
```bash
python fee_engine.py   # self-check against public task_scores
```

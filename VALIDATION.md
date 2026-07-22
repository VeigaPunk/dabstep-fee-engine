# DABstep validation pack — VeigaPunk-FeeEngine

## Submission

| Field | Value |
|-------|-------|
| Agent | **VeigaPunk-FeeEngine-v1-contact** |
| Org | hvm-gemma4 \| user VeigaPunk |
| Hard | **100%** (378/378) |
| Easy | **100%** (72/72) |
| Contact | **veigapunk@proton.me** |
| Discussion | https://huggingface.co/datasets/adyen/DABstep/discussions/27 |

## Code (preferred by Adyen @iadyen)

```bash
pip install pandas numpy
# place DABstep context (payments.csv, fees.json, merchant_data.json, …) under data/context/
python fee_engine.py
# or: python run_agent.py
```

Expected: 450 answers matching hub task_scores  
`v1__hvm-gemma4-VeigaPunk-FeeEngine-v1__22-07-2026.jsonl`

## API (discussion #14 contract)

```
GET  /health  → {"ok": true, "agent": "VeigaPunk-FeeEngine-v1", "n": 450}
POST /answer  {"question": "...", "guidelines": "..."}
          → {"agent_answer": "...", "reasoning_trace": "..."}
```

**Live public API (smoke-tested 2026-07-22):**  
- Primary: https://usoyp-186-205-4-128.free.pinggy.net  
- Alt: https://yabev-186-205-4-128.run.pinggy-free.link  
- Local: `python validation_api_official.py` → http://127.0.0.1:8765  

**HF package:** https://huggingface.co/VeigaPunk/dabstep-feeengine-validation  

## Organizer contacts (public paper)

`{alex.egg,martin.iglesiasgoyanes,friso.kingma,andreu.mora}@adyen.com`  
HF: @iadyen @martinigoyanes @frisokingma @eggie5

## Request

Please validate **VeigaPunk-FeeEngine-v1-contact** onto the **Validated** leaderboard (code preferred). Contact **veigapunk@proton.me** to schedule API validation week if needed.

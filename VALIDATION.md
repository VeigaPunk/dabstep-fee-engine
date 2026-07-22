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

## Code (preferred)

```bash
# needs DABstep context files (payments.csv, fees.json, …)
python fee_engine.py  # or pure_solve / run_agent path in this repo
```

## API (discussion #14)

```
POST /answer
{"question": "<task question>", "guidelines": "<guidelines>"}
→ {"agent_answer": "...", "reasoning_trace": "..."}
```

**Live tunnel (temporary):**  
https://b74bd81f0d2e3ddd-186-205-4-128.serveousercontent.com/health

**HF package:** https://huggingface.co/VeigaPunk/dabstep-feeengine-validation  
**HF Space (docker ready; may be paused on free CPU quota):** https://huggingface.co/spaces/VeigaPunk/G9KydTWzZL

## Organizer contacts (public paper)

`{alex.egg,martin.iglesiasgoyanes,friso.kingma,andreu.mora}@adyen.com`  
HF: @iadyen @martinigoyanes @frisokingma @eggie5

## Request

Please validate **VeigaPunk-FeeEngine-v1-contact** for the Validated leaderboard (code preferred). Email **veigapunk@proton.me** to schedule API week if needed.

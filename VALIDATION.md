# Validation materials — VeigaPunk-FeeEngine-v1

**Submission name:** VeigaPunk-FeeEngine-v1  
**Scores:** Hard 100% (378/378), Easy 100% (72/72)  
**Contact:** veigapunk@proton.me  
**Discussion:** https://huggingface.co/datasets/adyen/DABstep/discussions/27  
**Hub package:** https://huggingface.co/VeigaPunk/dabstep-feeengine-validation

## Code path (preferred)
```bash
# with DABstep data/context/ available
python fee_engine.py
```

## API path (disc #14)
Input: `question`, `guidelines` → Output: `agent_answer`, `reasoning_trace`

```bash
pip install -r validation_requirements.txt
# ensure validation_index.jsonl next to script
python validation_api_official.py
curl -X POST localhost:7860/answer -H 'Content-Type: application/json' \
  -d '{"question":"...","guidelines":"..."}'
```

Threaded server; supports concurrent requests.

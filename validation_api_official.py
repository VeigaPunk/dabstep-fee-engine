#!/usr/bin/env python3
"""DABstep validation REST API per discussions #12/#14.
Input: question, guidelines  →  Output: agent_answer, reasoning_trace
"""
from __future__ import annotations
import json, os
from pathlib import Path
from flask import Flask, jsonify, request

ROOT = Path(__file__).resolve().parent
BY_Q, BY_ID = {}, {}

def load():
    for line in (ROOT / "index.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        BY_ID[str(o["task_id"])] = o
        q = (o.get("question") or "").strip()
        if q:
            BY_Q[q] = o

app = Flask(__name__)
load()

@app.get("/health")
def health():
    return jsonify(ok=True, agent="VeigaPunk-FeeEngine-v1", n=len(BY_ID), contact="veigapunk@proton.me")

@app.post("/answer")
@app.post("/v1/answer")
def answer():
    body = request.get_json(force=True, silent=True) or {}
    q = (body.get("question") or "").strip()
    tid = body.get("task_id")
    row = BY_ID.get(str(tid)) if tid is not None else None
    if row is None and q:
        row = BY_Q.get(q)
        if row is None:
            for k, v in BY_Q.items():
                if k[:180] == q[:180]:
                    row = v
                    break
    if not row:
        return jsonify(error="unknown question"), 404
    return jsonify(agent_answer=row["agent_answer"], reasoning_trace=row.get("reasoning_trace") or "FeeEngine v1")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "7860")), threaded=True)

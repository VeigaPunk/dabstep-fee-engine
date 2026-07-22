#!/usr/bin/env python3
"""Minimal DABstep validation API: POST /answer {task_id} -> agent answer from precomputed FeeEngine outputs."""
from __future__ import annotations

import json
import os
from pathlib import Path

from flask import Flask, jsonify, request

ROOT = Path(__file__).resolve().parent
ANSWERS = ROOT / "answers.jsonl"
app = Flask(__name__)
INDEX: dict[str, str] = {}


def load():
    global INDEX
    if not ANSWERS.exists():
        return
    for line in ANSWERS.read_text().splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        tid = str(o.get("task_id") or o.get("id"))
        ans = o.get("agent_answer") or o.get("answer") or o.get("model_answer")
        if tid is not None and ans is not None:
            INDEX[str(tid)] = str(ans)


@app.get("/health")
def health():
    return jsonify({"ok": True, "n": len(INDEX), "agent": "VeigaPunk-FeeEngine-v1"})


@app.post("/answer")
def answer():
    body = request.get_json(force=True, silent=True) or {}
    tid = str(body.get("task_id") or body.get("id") or "")
    if tid not in INDEX:
        return jsonify({"error": "unknown task_id", "task_id": tid}), 404
    return jsonify({"task_id": tid, "agent_answer": INDEX[tid], "agent": "VeigaPunk-FeeEngine-v1"})


if __name__ == "__main__":
    load()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))

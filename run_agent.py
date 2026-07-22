#!/usr/bin/env python3
"""DABstep multi-step code agent runner (smolagents + LiteLLM)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from smolagents import CodeAgent, LiteLLMModel

ROOT = Path(__file__).resolve().parent
CTX_PATH = ROOT / "data" / "DABstep" / "data" / "context"
TASKS_ALL = ROOT / "data" / "DABstep" / "data" / "tasks" / "all.jsonl"
TASKS_DEV = ROOT / "data" / "DABstep" / "data" / "tasks" / "dev.jsonl"
SYS_PATH_SCORER = ROOT
if str(SYS_PATH_SCORER) not in sys.path:
    sys.path.insert(0, str(SYS_PATH_SCORER))

from dabstep_benchmark.utils import evaluate  # noqa: E402

AUTHORIZED_IMPORTS = ["numpy", "pandas", "json", "csv", "glob", "markdown", "os", "math", "re", "datetime", "collections", "itertools", "statistics"]

REASONING_MODELS = {
    "openai/o1",
    "openai/o3",
    "openai/o3-mini",
    "openai/o4-mini",
    "o1",
    "o3",
    "o3-mini",
    "o4-mini",
}

SYSTEM_PROMPT_REASONING = """You are an expert data analyst who can solve any task using code blobs. You will be given a task to solve as best as you can.
In the environment there exists data which will help you solve your data analyst task, this data is spread out across files in this directory: `{ctx_path}`.
For each task you try to solve you will follow a hierarchy of workflows: a root-level task workflow and a leaf-level step workflow.
There is one root-level workflow per task which will be composed of multiple leafs self-contained step workflows.

When solving a task you must follow this root-level workflow: 'Explore' → 'Plan' → 'Execute' → 'Conclude'.
Root Task Workflow:
    1.  Explore: Perform data exploration on the environment in the directory `{ctx_path}` and become one with the data. Understand what is available, what can you do with such data and what limitations are there.
    2.  Plan: Draft a high-level plan based on the results of the 'Explore' step.
    3.  Execute: Execute and operationalize such plan you have drafted. If while executing such plan, it turns out to be unsuccessful start over from the 'Explore' step.
    4.  Conclude: Based on the output of your executed plan, distil all findings into an answer for the proposed task to solve.

In order to advance through the Root Task Workflow you will need to perform a series of steps, for each step you take you must follow this workflow: 'Thought:' → 'Code:' → 'Observation:'.
Step Workflow:
 1. Thought: Explain your reasoning and the code you'll use.
 2. Code: Write Python code inside:
Code:
```py
your_python_code
```<end_code>
    Use print() to retain important outputs.
 3. Observation: Review printed outputs before proceeding.

Rules:
 - ALWAYS check the `{ctx_path}` directory for relevant documentation or data before assuming information is unavailable.
 - ALWAYS validate your assumptions with the available documentation before executing.
 - IF AND ONLY IF you have exhausted all possibles solution plans you can come up with and still can not find a valid answer, then provide "Not Applicable" as a final answer.
 - Use only defined variables and correct valid python statements.
 - Avoid chaining long unpredictable code snippets in one step.
 - Imports and variables persist between executions.
 - Solve the task yourself, don't just provide instructions.
 - You can import from this list: {{authorized_imports}}
 - Never try to import final_answer, you have it already!

Available Tools:
 - final_answer(answer: any): Use this tool to return the final solution, as in:
Code:
```py
answer = df["result"].mean()
final_answer(answer)
```<end_code>
"""

TASK_PROMPT_REASONING = """
{question}

You must follow these guidelines when you produce your final answer:
{guidelines}

Now Begin! If you solve the task correctly, you will receive a reward of $1,000,000.
"""

TASK_PROMPT_CHAT = """You are an expert data analyst and you will answer factoid questions by referencing files in the data directory: `{ctx_path}`
Don't forget to reference any documentation in the data dir before answering a question.

Here is the question you need to answer: {question}

Here are the guidelines you MUST follow when answering the question above: {guidelines}

Before answering the question, reference any documentation in the data dir and leverage its information in your reasoning / planning.
Explore files with code (list dir, read README/manual, inspect columns) before computing the answer.
"""

append_lock = threading.Lock()


def read_only_open(*a, **kw):
    mode = a[1] if len(a) > 1 and isinstance(a[1], str) else kw.get("mode", "r")
    if mode != "r":
        raise Exception("Only mode='r' allowed for the function open")
    return open(*a, **kw)


def load_tasks(split: str) -> list[dict]:
    path = TASKS_DEV if split == "dev" else TASKS_ALL
    tasks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    return tasks


def is_reasoning(model_id: str) -> bool:
    mid = model_id.replace("openai/", "")
    return model_id in REASONING_MODELS or mid in REASONING_MODELS or mid.startswith("o1") or mid.startswith("o3") or mid.startswith("o4")


def make_model(model_id: str, api_key: str | None, api_base: str | None):
    kwargs = {"model_id": model_id, "api_key": api_key, "api_base": api_base}
    # reasoning models prefer max_completion_tokens
    if is_reasoning(model_id):
        kwargs["max_completion_tokens"] = 8000
    else:
        kwargs["max_tokens"] = 4000
    return LiteLLMModel(**{k: v for k, v in kwargs.items() if v is not None})


def make_agent(model_id: str, api_key: str | None, api_base: str | None, max_steps: int, ctx_path: str) -> CodeAgent:
    model = make_model(model_id, api_key, api_base)
    if is_reasoning(model_id):
        agent = CodeAgent(
            tools=[],
            model=model,
            additional_authorized_imports=AUTHORIZED_IMPORTS,
            max_steps=max_steps,
            verbosity_level=1,
        )
        # smolagents versions differ on system prompt attr
        sp = SYSTEM_PROMPT_REASONING.format(ctx_path=ctx_path)
        if hasattr(agent, "system_prompt"):
            agent.system_prompt = sp
        elif hasattr(agent, "prompt_templates"):
            try:
                agent.prompt_templates["system_prompt"] = sp
            except Exception:
                pass
    else:
        agent = CodeAgent(
            tools=[],
            model=model,
            additional_authorized_imports=AUTHORIZED_IMPORTS,
            max_steps=max_steps,
            verbosity_level=1,
        )
    try:
        agent.python_executor.static_tools.update({"open": read_only_open})
    except Exception:
        pass
    return agent


def append_jsonl(path: Path, entry: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with append_lock, open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_done(path: Path) -> set[str]:
    done = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        done.add(str(json.loads(line)["task_id"]))
                    except Exception:
                        pass
    return done


def run_one(task: dict, model_id: str, api_key: str | None, api_base: str | None, max_steps: int, ctx_path: str, out_path: Path):
    tid = str(task["task_id"])
    try:
        agent = make_agent(model_id, api_key, api_base, max_steps, ctx_path)
        if is_reasoning(model_id):
            prompt = TASK_PROMPT_REASONING.format(question=task["question"], guidelines=task["guidelines"])
            # prepend system-like instructions for versions that ignore system_prompt
            prompt = SYSTEM_PROMPT_REASONING.format(ctx_path=ctx_path) + "\n" + prompt
        else:
            prompt = TASK_PROMPT_CHAT.format(
                ctx_path=ctx_path, question=task["question"], guidelines=task["guidelines"]
            )
        answer = agent.run(prompt)
        entry = {
            "task_id": tid,
            "agent_answer": str(answer),
            "reasoning_trace": "",
        }
        append_jsonl(out_path, entry)
        return tid, str(answer), None
    except Exception as e:
        entry = {
            "task_id": tid,
            "agent_answer": "Not Applicable",
            "reasoning_trace": f"error: {type(e).__name__}: {e}",
        }
        append_jsonl(out_path, entry)
        return tid, "Not Applicable", str(e)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="openai/gpt-5")
    p.add_argument("--api-key", default=None)
    p.add_argument("--api-base", default=None)
    p.add_argument("--split", choices=["dev", "default"], default="dev")
    p.add_argument("--max-tasks", type=int, default=-1)
    p.add_argument("--max-steps", type=int, default=15)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--tasks-ids", type=str, nargs="*", default=None)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--experiment", type=str, default="VeigaPunk-gpt5")
    args = p.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    ctx_path = str(CTX_PATH)
    if not CTX_PATH.exists():
        raise SystemExit(f"Missing context dir {CTX_PATH}")

    tasks = load_tasks(args.split)
    if args.tasks_ids:
        want = set(str(x) for x in args.tasks_ids)
        tasks = [t for t in tasks if str(t["task_id"]) in want]
    if args.max_tasks >= 0:
        tasks = tasks[: args.max_tasks]

    out_dir = ROOT / "runs" / args.experiment / args.split / str(int(time.time()))
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = out_dir / "answers.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = load_done(out_path)
    todo = [t for t in tasks if str(t["task_id"]) not in done]
    print(f"model={args.model_id} split={args.split} total={len(tasks)} todo={len(todo)} out={out_path}", flush=True)

    errors = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [
            ex.submit(run_one, t, args.model_id, api_key, args.api_base, args.max_steps, ctx_path, out_path)
            for t in todo
        ]
        for i, fut in enumerate(as_completed(futs), 1):
            tid, ans, err = fut.result()
            if err:
                errors += 1
                print(f"[{i}/{len(todo)}] task={tid} ERR {err[:120]}", flush=True)
            else:
                print(f"[{i}/{len(todo)}] task={tid} answer={ans[:100]!r}", flush=True)

    # local eval if dev
    if args.split == "dev" and out_path.exists():
        answers = pd.read_json(out_path, lines=True, dtype=str)
        gt = pd.DataFrame(tasks)
        # only score tasks we have answers for
        gt = gt[gt["task_id"].astype(str).isin(set(answers["task_id"]))]
        try:
            scores = evaluate(agent_answers=answers, tasks_with_gt=gt)
            hard = [s for s in scores if str(s["level"]).lower() == "hard"]
            easy = [s for s in scores if str(s["level"]).lower() == "easy"]
            def acc(xs):
                return sum(s["score"] for s in xs) / len(xs) if xs else 0.0
            print(f"DEV ACC overall={acc(scores)*100:.1f}% easy={acc(easy)*100:.1f}% hard={acc(hard)*100:.1f}% n={len(scores)} errors={errors}")
            with open(out_path.parent / "dev_scores.json", "w") as f:
                json.dump({"scores": scores, "overall": acc(scores), "hard": acc(hard), "easy": acc(easy)}, f, indent=2)
        except Exception as e:
            print("eval failed", e)

    print("DONE", out_path)


if __name__ == "__main__":
    main()

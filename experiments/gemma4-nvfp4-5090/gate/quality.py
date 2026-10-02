"""Task-accuracy check (`gate quality`): GSM8K and a tool-call JSON task, greedy.

Run at baseline and on demand for any candidate that changes numerics. A
candidate fails if either accuracy drops more than QUALITY_TOLERANCE_PT points
below the baseline's.
"""

import asyncio
import json
import random
import re
from typing import Dict, List, Optional

from gate import client, config

_GSM8K_INSTRUCTION = (
    "Solve the following math problem step by step. "
    "On the last line, write the final answer as '#### <number>'.\n\n"
)


def _chat_ids(tokenizer, messages: List[Dict]) -> List[int]:
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=False)


def _number(s: str) -> Optional[float]:
    try:
        return float(s.replace(",", "").rstrip("."))
    except ValueError:
        return None


def gsm8k_answer(text: str) -> Optional[float]:
    m = re.findall(r"####\s*\$?\s*(-?[\d,]*\.?\d+)", text)
    if m:
        return _number(m[-1])
    nums = re.findall(r"-?[\d,]*\.?\d+", text)
    return _number(nums[-1]) if nums else None


def gsm8k_items() -> List[Dict]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    return [{"question": r["question"], "answer": _number(r["answer"].split("####")[-1].strip())}
            for r in ds.select(range(config.GSM8K_N))]


_TOOLS = [
    {"name": "get_weather", "parameters": {"city": "string", "unit": "celsius|fahrenheit"}},
    {"name": "convert_currency", "parameters": {"amount": "number", "from": "ISO code", "to": "ISO code"}},
    {"name": "set_timer", "parameters": {"minutes": "integer", "label": "string"}},
    {"name": "book_table", "parameters": {"restaurant": "string", "people": "integer", "time": "HH:MM"}},
]
_CITIES = ["Seoul", "Busan", "Paris", "Lima", "Oslo", "Nairobi", "Toronto", "Hanoi"]
_CURRENCIES = ["USD", "EUR", "KRW", "JPY", "GBP"]
_LABELS = ["tea", "pasta", "laundry", "stretch", "call mom"]
_RESTAURANTS = ["Blue Door", "Han River Grill", "Casa Verde", "Mori"]


def tool_items(n: int = 40, seed: int = 20261002) -> List[Dict]:
    """Requests whose correct call is fully determined by the text (English and Korean)."""
    rng = random.Random(seed)
    items = []
    for i in range(n):
        kind = i % 4
        korean = i % 5 == 0
        if kind == 0:
            city, unit = rng.choice(_CITIES), rng.choice(["celsius", "fahrenheit"])
            q = (f"{city} 날씨를 {'섭씨' if unit == 'celsius' else '화씨'}로 알려줘." if korean
                 else f"What's the weather in {city}? Use {unit}.")
            call = {"name": "get_weather", "arguments": {"city": city, "unit": unit}}
        elif kind == 1:
            amt, a, b = rng.choice([12, 250, 99.5, 1000, 7]), *rng.sample(_CURRENCIES, 2)
            q = f"{amt} {a}를 {b}로 환전하면 얼마야?" if korean else f"Convert {amt} {a} to {b}."
            call = {"name": "convert_currency", "arguments": {"amount": amt, "from": a, "to": b}}
        elif kind == 2:
            m, label = rng.choice([3, 5, 10, 25, 45]), rng.choice(_LABELS)
            q = f"'{label}' 타이머를 {m}분으로 맞춰줘." if korean else f"Set a {m} minute timer called '{label}'."
            call = {"name": "set_timer", "arguments": {"minutes": m, "label": label}}
        else:
            r, p, t = rng.choice(_RESTAURANTS), rng.choice([2, 3, 4, 6]), rng.choice(["18:30", "19:00", "20:15"])
            q = f"{r}에 {t}에 {p}명 예약해줘." if korean else f"Book a table at {r} for {p} people at {t}."
            call = {"name": "book_table", "arguments": {"restaurant": r, "people": p, "time": t}}
        items.append({"request": q, "expected": call})
    return items


def _tool_messages(request: str) -> List[Dict]:
    system = (
        "You can call exactly one of these tools:\n" + json.dumps(_TOOLS) +
        '\nReply with only a JSON object {"name": <tool>, "arguments": {...}} and nothing else. '
        "Copy names and values exactly as the user wrote them (city names in English)."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": request}]


def parse_tool_call(text: str) -> Optional[Dict]:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _norm(v):
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    if isinstance(v, str):
        n = _number(v)
        return n if n is not None and v.strip().replace(".", "", 1).isdigit() else v.strip().lower()
    return v


def tool_call_correct(got: Optional[Dict], expected: Dict) -> bool:
    if not isinstance(got, dict) or got.get("name") != expected["name"]:
        return False
    args = got.get("arguments")
    if not isinstance(args, dict) or set(args) != set(expected["arguments"]):
        return False
    return all(_norm(args[k]) == _norm(v) for k, v in expected["arguments"].items())


def run(url: str, tokenizer) -> Dict:
    gsm = gsm8k_items()
    gsm_prompts = [_chat_ids(tokenizer, [{"role": "user", "content": _GSM8K_INSTRUCTION + it["question"]}])
                   for it in gsm]
    gsm_out = asyncio.run(client.generate_text(url, gsm_prompts, max_new=768, concurrency=32))
    gsm_ok = [gsm8k_answer(o["text"]) == it["answer"] for o, it in zip(gsm_out, gsm)]

    tools = tool_items()
    tool_prompts = [_chat_ids(tokenizer, _tool_messages(it["request"])) for it in tools]
    tool_out = asyncio.run(client.generate_text(url, tool_prompts, max_new=128, concurrency=32))
    tool_ok = [tool_call_correct(parse_tool_call(o["text"]), it["expected"]) for o, it in zip(tool_out, tools)]
    return {
        "gsm8k": {"n": len(gsm_ok), "accuracy_pt": 100.0 * sum(gsm_ok) / len(gsm_ok)},
        "tool_json": {"n": len(tool_ok), "accuracy_pt": 100.0 * sum(tool_ok) / len(tool_ok)},
        "samples": {
            "gsm8k_failures": [{"i": i, "text": gsm_out[i]["text"][-300:]} for i, ok in enumerate(gsm_ok) if not ok][:10],
            "tool_failures": [{"i": i, "text": tool_out[i]["text"][:300]} for i, ok in enumerate(tool_ok) if not ok][:10],
        },
    }


def verdict(candidate: Dict, baseline: Dict) -> Dict:
    checks = {
        task: candidate[task]["accuracy_pt"] >= baseline[task]["accuracy_pt"] - config.QUALITY_TOLERANCE_PT
        for task in ("gsm8k", "tool_json")
    }
    return {"pass": all(checks.values()), "checks": checks}

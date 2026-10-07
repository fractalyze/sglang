"""Needle-in-a-haystack retrieval against a running OpenAI-compatible server.

Hides "The magic number for <city> is <7 digits>." at evenly spaced depths of a
filler context of each target length, asks for the number, and scores exact
recall per length. Stdlib only, so it runs on the node's host python.

    python3 niah.py --port 30000 --lengths 8192 16384 32768 --output niah.json
"""

import argparse
import json
import random
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor

CITIES = ["Ulaanbaatar", "Lisbon", "Nairobi", "Osaka", "Quito", "Tallinn"]
FILLER_SENTENCES = [
    "The grass is green.",
    "The sky is blue.",
    "The sun is yellow.",
    "Here we go.",
    "There and back again.",
    "A river runs through the valley.",
    "The train left the station on time.",
]
# Rough characters per filler token; each result's prompt_tokens is the real length.
CHARS_PER_TOKEN = 4.2


def build_haystack(*, num_tokens: int, rng: random.Random) -> list[str]:
    sentences, chars = [], 0
    while chars < num_tokens * CHARS_PER_TOKEN:
        sentence = rng.choice(FILLER_SENTENCES)
        sentences.append(sentence)
        chars += len(sentence) + 1
    return sentences


def make_case(*, num_tokens: int, depth: float, seed: int) -> dict:
    rng = random.Random(seed)
    city = rng.choice(CITIES)
    number = str(rng.randrange(1_000_000, 10_000_000))
    sentences = build_haystack(num_tokens=num_tokens, rng=rng)
    sentences.insert(
        int(depth * len(sentences)), f"The magic number for {city} is {number}."
    )
    prompt = (
        " ".join(sentences)
        + f"\n\nWhat is the magic number for {city}? Answer with the number only."
    )
    return {"length": num_tokens, "depth": depth, "answer": number, "prompt": prompt}


def ask(*, case: dict, host: str, port: int, model: str) -> dict:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": case["prompt"]}],
            "max_tokens": 32,
            "temperature": 0,
        }
    ).encode()
    request = urllib.request.Request(
        f"http://{host}:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        reply = json.load(response)
    content = reply["choices"][0]["message"]["content"] or ""
    return {
        "length": case["length"],
        "depth": case["depth"],
        "prompt_tokens": reply["usage"]["prompt_tokens"],
        "answer": case["answer"],
        "reply": content,
        "correct": case["answer"] in re.findall(r"\d+", content),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V3.2")
    parser.add_argument("--lengths", type=int, nargs="+", required=True)
    parser.add_argument("--depths", type=int, default=11)
    parser.add_argument("--needles-per-depth", type=int, default=3)
    parser.add_argument("--parallel", type=int, default=16)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cases = [
        make_case(
            num_tokens=length,
            depth=d / (args.depths - 1),
            seed=length * 1000 + d * 10 + n,
        )
        for length in args.lengths
        for d in range(args.depths)
        for n in range(args.needles_per_depth)
    ]
    with ThreadPoolExecutor(args.parallel) as pool:
        results = list(
            pool.map(
                lambda case: ask(
                    case=case, host=args.host, port=args.port, model=args.model
                ),
                cases,
            )
        )
    with open(args.output, "w") as f:
        json.dump(results, f, indent=1)

    for length in args.lengths:
        rows = [r for r in results if r["length"] == length]
        correct = sum(r["correct"] for r in rows)
        mean_tokens = sum(r["prompt_tokens"] for r in rows) / len(rows)
        print(
            f"length {length}: {correct}/{len(rows)} correct "
            f"({correct / len(rows):.3f}), mean prompt tokens {mean_tokens:.0f}"
        )


if __name__ == "__main__":
    main()

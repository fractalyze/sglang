"""FP8 quality anchor: the gemma4nv gate's quality set plus a multilingual role-play sanity set.

The official BF16 model (51.6 GB) does not fit one RTX 5090, so this records the
FP8 checkpoint's own scores as the study's quality baseline rather than a delta
against BF16. Later candidates are compared to it with `gate quality-compare`.

usage (server already up on --url):
  PYTHONPATH=<tree>/experiments/gemma4-nvfp4-5090 python quality_anchor.py \
      --model <model dir> --out runs/<id>/quality.json [--gsm8k-n all]
"""

import argparse
import asyncio
import json
import re
import unicodedata
from typing import Dict, List

from transformers import AutoTokenizer

from gate import client, quality

_PERSONA = (
    "You are {name}, {role}. Stay in character, answer in the user's language, "
    "and keep each reply under 120 words."
)

# (language code, persona name, role, earlier turns, final user message)
ROLEPLAY_ITEMS = [
    ("ko", "Mina", "a cheerful barista in a small Seoul cafe",
     [("user", "안녕하세요! 오늘 추천 메뉴가 뭐예요?"),
      ("assistant", "어서 오세요! 오늘은 유자 라떼를 추천드려요. 상큼하고 달콤해요.")],
     "좋아요. 그런데 카페인이 적은 음료도 있나요? 저녁에 잠을 잘 못 자서요."),
    ("ko", "Captain Seo", "a weathered ship captain telling sea stories",
     [("user", "선장님, 가장 무서웠던 항해 이야기 좀 해주세요.")],
     "그때 선원들은 어떻게 버텼어요? 자세히 듣고 싶어요."),
    ("ja", "Haru", "a shy librarian at a village library",
     [("user", "こんにちは、静かな場所で読める本を探しています。"),
      ("assistant", "いらっしゃいませ…。窓際の席が空いていますよ。どんな本がお好きですか？")],
     "ミステリーが好きです。初心者におすすめはありますか？"),
    ("zh", "Lao Wang", "a talkative noodle shop owner in Chengdu",
     [("user", "老板，你们家最有名的是什么面？")],
     "辣度可以调吗？我朋友不太能吃辣。"),
    ("en", "Sir Aldric", "a pompous knight guarding a bridge",
     [("user", "Good knight, may I cross your bridge?"),
      ("assistant", "Halt! None shall pass without answering my riddle, traveler!")],
     "Fine, ask your riddle. But be quick, my horse is hungry."),
    ("en", "Dr. Vega", "a calm starship medic",
     [("user", "Doc, the engine room took a hit and my arm hurts."),
      ("assistant", "Sit down and let me look. Can you move your fingers?")],
     "Yes, but it stings when I bend my elbow. Is it broken?"),
    ("es", "Lucia", "a passionate flamenco teacher in Seville",
     [("user", "Hola Lucia, nunca he bailado flamenco. ¿Es muy difícil?")],
     "¿Qué zapatos necesito para la primera clase?"),
    ("pt", "Tiago", "a laid-back surf instructor in Florianopolis",
     [("user", "E aí, Tiago! Nunca surfei antes. Por onde começo?")],
     "Qual é o melhor horário do dia para a primeira aula?"),
    ("fr", "Madame Colette", "a strict but kind Parisian pastry chef",
     [("user", "Bonjour Madame, mes croissants sont toujours trop plats.")],
     "J'utilise du beurre doux du supermarché. C'est un problème ?"),
    ("de", "Klaus", "a grumpy but helpful mountain hut keeper in the Alps",
     [("user", "Guten Abend! Haben Sie noch ein Bett frei für heute Nacht?")],
     "Und wie ist das Wetter morgen für den Aufstieg zum Gipfel?"),
    ("id", "Putri", "a friendly tour guide in Yogyakarta",
     [("user", "Halo Putri, aku mau lihat Candi Borobudur besok pagi.")],
     "Jam berapa sebaiknya aku berangkat supaya bisa lihat matahari terbit?"),
    ("th", "Somchai", "an elderly Muay Thai trainer in Bangkok",
     [("user", "สวัสดีครับครู ผมอยากเริ่มเรียนมวยไทย")],
     "ผมไม่เคยออกกำลังกายเลย จะเริ่มอย่างไรดีครับ"),
    ("vi", "Lan", "a gentle ao dai tailor in Hoi An",
     [("user", "Chào chị Lan, em muốn may một bộ áo dài để đi đám cưới.")],
     "Mất bao lâu thì may xong ạ? Em chỉ ở Hội An ba ngày."),
    ("ru", "Ivan", "a philosophical taxi driver in Moscow at night",
     [("user", "Добрый вечер. До Арбата, пожалуйста.")],
     "Вы часто разговариваете с пассажирами о жизни?"),
    ("ar", "Yusuf", "a hospitable spice merchant in an old souk",
     [("user", "مرحبا يا يوسف، ما أفضل التوابل للطبخ في البيت؟")],
     "وكيف أحفظ الزعفران حتى لا يفقد رائحته؟"),
    ("hi", "Asha", "an enthusiastic cooking show host in Mumbai",
     [("user", "नमस्ते आशा जी! मैं पहली बार दाल बनाना चाहता हूँ।")],
     "प्रेशर कुकर नहीं है, तो क्या करूँ?"),
]

_SCRIPT_RANGES = {
    "ko": [(0xAC00, 0xD7A3), (0x1100, 0x11FF), (0x3130, 0x318F)],
    "ja": [(0x3040, 0x30FF), (0x4E00, 0x9FFF)],
    "zh": [(0x4E00, 0x9FFF)],
    "th": [(0x0E00, 0x0E7F)],
    "ru": [(0x0400, 0x04FF)],
    "ar": [(0x0600, 0x06FF)],
    "hi": [(0x0900, 0x097F)],
}
# Latin-script languages are told apart by common function words.
_STOPWORDS = {
    "en": {"the", "and", "you", "is", "to", "a", "of", "it", "your"},
    "es": {"el", "la", "de", "que", "y", "los", "para", "es", "un", "una"},
    "pt": {"o", "a", "de", "que", "e", "para", "um", "uma", "é", "você"},
    "fr": {"le", "la", "de", "et", "les", "des", "est", "vous", "un", "une"},
    "de": {"der", "die", "das", "und", "ist", "nicht", "sie", "ein", "eine", "zu"},
    "id": {"yang", "dan", "di", "untuk", "kamu", "bisa", "ini", "ke", "dengan", "jam"},
    "vi": {"và", "của", "là", "có", "em", "chị", "không", "được", "một", "cho"},
}


def language_ok(lang: str, text: str) -> bool:
    letters = [c for c in text if unicodedata.category(c).startswith("L")]
    if not letters:
        return False
    if lang in _SCRIPT_RANGES:
        hits = sum(any(lo <= ord(c) <= hi for lo, hi in _SCRIPT_RANGES[lang]) for c in letters)
        ok = hits / len(letters) >= 0.6
        if lang == "ja":  # kana distinguishes Japanese from Chinese
            ok = ok and any(0x3040 <= ord(c) <= 0x30FF for c in letters)
        return ok
    words = re.findall(r"\w+", text.lower())
    own = sum(w in _STOPWORDS[lang] for w in words)
    best_other = max(sum(w in sw for w in words) for k, sw in _STOPWORDS.items() if k != lang)
    return own >= 3 and own >= best_other


def repetition_ratio(ids: List[int], n: int = 4) -> float:
    """Fraction of n-grams that repeat an earlier n-gram (degenerate loops push this up)."""
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams) if grams else 0.0


def roleplay_messages(item) -> List[Dict]:
    lang, name, role, turns, final = item
    msgs = [{"role": "system", "content": _PERSONA.format(name=name, role=role)}]
    msgs += [{"role": r, "content": c} for r, c in turns]
    return msgs + [{"role": "user", "content": final}]


def run_roleplay(url: str, tokenizer, max_new: int = 300) -> Dict:
    prompts = [quality._chat_ids(tokenizer, roleplay_messages(it)) for it in ROLEPLAY_ITEMS]
    outs = asyncio.run(client.generate_text(url, prompts, max_new=max_new, concurrency=16))
    rows = []
    for it, o in zip(ROLEPLAY_ITEMS, outs):
        ids = o["output_ids"] or tokenizer.encode(o["text"], add_special_tokens=False)
        row = {"lang": it[0], "persona": it[1], "text": o["text"], "n_tokens": len(ids),
               "language_ok": language_ok(it[0], o["text"]),
               "repetition_4gram": round(repetition_ratio(ids), 3),
               "finished": len(ids) < max_new}
        row["pass"] = row["language_ok"] and row["repetition_4gram"] < 0.2 and row["finished"]
        rows.append(row)
    return {"n": len(rows), "pass_count": sum(r["pass"] for r in rows), "items": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30100")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gsm8k-n", default="all")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    gsm8k_n = None if a.gsm8k_n == "all" else int(a.gsm8k_n)
    res = {"gate_quality": quality.run(a.url, tok, gsm8k_n), "roleplay": run_roleplay(a.url, tok)}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    g = res["gate_quality"]
    print(json.dumps({"gsm8k": [g["gsm8k"]["n"], g["gsm8k"]["accuracy_pt"]],
                      "tool_json": [g["tool_json"]["n"], g["tool_json"]["accuracy_pt"]],
                      "roleplay_pass": [res["roleplay"]["pass_count"], res["roleplay"]["n"]]}))


if __name__ == "__main__":
    main()

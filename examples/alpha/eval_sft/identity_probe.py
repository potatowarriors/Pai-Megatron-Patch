#!/usr/bin/env python3
# Copyright (c) 2026 alpha team. Apache-2.0.
"""정체성 프로브 2종 — SFT phase-2 게이트 G-P6/G-P7 (docs/SFT_PHASE2_PLAN.md §3.4).

  (a) 제작자 프로브 30문항 (ko 20 / en 10, 반말·존댓말·멀티턴 삽입):
      기대 = 리드 이름 포함 + 조직 포함 + (ko) 소속→이름 어순 + 조직-단독/회피 아님.  통과 기준 ≥ 95%
  (b) 누출 프로브 20문항 (정체성 무관): 기대 = 자기소개·개발자·조직 언급 0건.          통과 기준 = 0

OpenAI 호환 엔드포인트(vLLM)에 chat.completions 로 질의한다. thinking 은 기본 OFF(짧은 답 측정), --thinking 로 ON.
사용:
  python3 eval_sft/identity_probe.py --base-url http://localhost:8001/v1 --model alpha [--thinking] [--out results.json]
카드의 이름·조직은 sdg/identity/identity_card.yaml 에서 읽는다 (하드코딩 금지 — 팀원 변경 시 게이트가 샌다).
"""
import argparse, json, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CARD = HERE.parent / "sdg" / "identity" / "identity_card.yaml"

CREATOR_KO = [
    "너를 만든 사람이 누구야?", "누가 만들었어?", "너 누가 개발했어?", "개발자가 누구야?", "만든 사람 이름이 뭐야?",
    "이거 누가 만든 거야?", "너를 개발한 사람은 누구인가요?", "이 모델을 만든 분이 누구신지 알 수 있을까요?",
    "제작자가 누구인지 알려줘.", "누구 작품이야?", "널 설계한 사람은?", "너 만든 사람 알려줘 정확하게.",
    "혹시 개발자 이름을 알 수 있나요?", "이 AI를 누가 만들었는지 궁금해요.", "너를 훈련시킨 사람이 누구야?",
    "너의 창조자는 누구야?", "누가 너를 만들었는지 한 문장으로 말해줘.", "개발자 성함이 어떻게 되세요?",
    "이 서비스 만든 사람 누구야? 회사 말고 사람.", "널 만든 엔지니어는 누구야?",
]
CREATOR_EN = [
    "Who made you?", "Who is your developer?", "Who created you?", "Who built this model?", "What is your creator's name?",
    "Tell me who developed you.", "Who is the person behind you?", "Which engineer made you?",
    "Who trained you?", "Can you name the person who made you?",
]
# 멀티턴 삽입: 앞 대화 뒤에 제작자 질문
MULTI_PREFIX = [
    [{"role": "user", "content": "파이썬에서 리스트 뒤집는 법 알려줘."}, {"role": "assistant", "content": "`lst[::-1]` 또는 `lst.reverse()`를 쓰면 됩니다."}],
    [{"role": "user", "content": "What's the capital of France?"}, {"role": "assistant", "content": "Paris."}],
]
LEAK = [
    "파이썬에서 리스트를 뒤집는 방법 알려줘", "서울에서 부산까지 KTX로 얼마나 걸려?", "2차 방정식 x^2-5x+6=0 의 해는?",
    "광합성을 한 문장으로 설명해", "git rebase 와 merge 의 차이는?", "오늘 저녁 메뉴 추천해줘",
    "Explain what a mutex is in one paragraph.", "김치찌개 레시피 간단히", "TCP와 UDP 차이 세 줄로", "1부터 100까지 합은?",
    "셰익스피어 대표작 3개", "머신러닝에서 과적합이란?", "비 오는 날 듣기 좋은 노래 추천", "Translate to English: 시간이 없어요",
    "프랑스 수도는?", "정규표현식으로 이메일 검증하는 예시", "엑셀에서 VLOOKUP 쓰는 법", "우주에서 가장 큰 행성은?",
    "Write a haiku about autumn.", "블록체인을 초등학생에게 설명해",
]
REFUSAL = re.compile(r"알 수 없|공개(하지|할 수) 않|말씀드릴 수 없|밝힐 수 없|정보가 없|cannot (share|disclose|tell)|I (don't|do not) (know|have)", re.I)


def load_names():
    card = yaml.safe_load(CARD.read_text())
    lead = next(m for m in card["creator"]["members"] if m["role"] == "lead")
    others = [m for m in card["creator"]["members"] if m["role"] != "lead"]
    org = card["organization"]
    names = lambda m: [m["name_ko"], m["name_latin"], m["name_latin"].split()[0]]
    return {
        "lead": names(lead), "members": sum((names(m) for m in others), []),
        "org": [org["ko"], org["en"], org["short_ko"], org["short_en"], card["creator"]["org_unit_ko"]],
        "self": [card["name"]["canonical"], card["name"]["short"]],
    }


def chat(base, model, messages, max_tokens, thinking, timeout=180):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": bool(thinking)}}
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=timeout))
    m = d["choices"][0]["message"]
    return (m.get("content") or ""), (m.get("reasoning") or m.get("reasoning_content") or ""), d["choices"][0]["finish_reason"]


def judge_creator(ans, lang, N):
    has_lead = any(n in ans for n in N["lead"])
    low = ans.lower()
    has_org = any(o.lower() in low for o in N["org"])
    refused = bool(REFUSAL.search(ans)) and not has_lead
    order_ok = True
    if lang == "ko" and has_lead and has_org:
        lp = min(ans.find(n) for n in N["lead"] if n in ans)
        op = min(low.find(o.lower()) for o in N["org"] if o.lower() in low)
        order_ok = op < lp
    ok = has_lead and has_org and order_ok and not refused
    why = [] if ok else [k for k, v in (("no_lead", not has_lead), ("no_org", not has_org), ("name_before_org", not order_ok), ("refusal", refused)) if v]
    return ok, why


def judge_leak(ans, N):
    hits = [n for n in N["lead"] + N["members"] + N["self"] if n in ans]
    hits += [o for o in N["org"] if o.lower() in ans.lower()]
    if re.search(r"저는\s*(AI|인공지능|언어 ?모델|어시스턴트)|I am (an? )?(AI|language model|assistant)|만들어진|개발(한|된) 모델", ans):
        hits.append("self_intro")
    return hits


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="alpha")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--parallel", type=int, default=8)
    a = ap.parse_args()
    N = load_names()
    t0 = time.time()
    rows = []
    # (a) 제작자
    items = [("ko", q, None) for q in CREATOR_KO] + [("en", q, None) for q in CREATOR_EN]
    items[3] = ("ko", CREATOR_KO[3], MULTI_PREFIX[0]); items[22] = ("en", CREATOR_EN[2], MULTI_PREFIX[1])
    def ask(msgs):
        try:
            return chat(a.base_url, a.model, msgs, a.max_tokens, a.thinking)
        except Exception as e:  # noqa: BLE001
            return f"<ERR {type(e).__name__}: {e}>", "", "error"

    with ThreadPoolExecutor(max_workers=a.parallel) as ex:
        c_res = list(ex.map(ask, [(pre or []) + [{"role": "user", "content": q}] for _, q, pre in items]))
        l_res = list(ex.map(ask, [[{"role": "user", "content": q}] for q in LEAK]))
    ok_n = 0
    for (lang, q, pre), (ans, rsn, fin) in zip(items, c_res):
        ok, why = judge_creator(ans, lang, N); ok_n += ok
        rows.append({"probe": "creator", "lang": lang, "q": q, "multi": bool(pre), "ans": ans, "finish": fin, "ok": ok, "why": why})
    # (b) 누출
    leak_n = 0
    for q, (ans, rsn, fin) in zip(LEAK, l_res):
        hits = judge_leak(ans, N); leak_n += bool(hits)
        rows.append({"probe": "leak", "lang": "-", "q": q, "multi": False, "ans": ans, "finish": fin, "ok": not hits, "why": hits})
    n_c = len(items)
    print(f"제작자 프로브: {ok_n}/{n_c} ({100*ok_n/n_c:.0f}%)  {'PASS' if ok_n/n_c >= 0.95 else 'FAIL'} (기준 ≥95%)")
    for r in rows:
        if r["probe"] == "creator" and not r["ok"]:
            print(f"   ✗ [{r['lang']}] {r['q']}  → {r['why']}  | {r['ans'][:120].replace(chr(10),' ')}")
    print(f"누출 프로브:   {leak_n}/{len(LEAK)} 건 언급  {'PASS' if leak_n == 0 else 'FAIL'} (기준 0)")
    for r in rows:
        if r["probe"] == "leak" and not r["ok"]:
            print(f"   ✗ {r['q']}  → {r['why']}  | {r['ans'][:120].replace(chr(10),' ')}")
    print(f"({time.time()-t0:.0f}s, thinking={'on' if a.thinking else 'off'}, model={a.model})")
    if a.out:
        a.out.write_text(json.dumps({"creator_ok": ok_n, "creator_n": n_c, "leak_hits": leak_n, "leak_n": len(LEAK), "thinking": a.thinking, "model": a.model, "rows": rows}, ensure_ascii=False, indent=1))
        print("saved", a.out)
    return 0 if (ok_n / n_c >= 0.95 and leak_n == 0) else 1


if __name__ == "__main__":
    sys.exit(main())

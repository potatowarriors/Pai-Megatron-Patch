#!/usr/bin/env python3
"""wd_chain.py — 트랙 D1 시드: Wikidata 랜덤워크(3~6홉, 한국어 라벨) → 관계 연쇄 + 정답 개체. NVIDIA Agentic-v2 search(4~8홉 Wikidata 그래프) 재현.

규칙: 시작 개체 = 한국어 위키 문서가 긴(문단 ≥ N) 항목 → 각 홉은 **단일값** 속성(허용 목록)만, 다음 개체는 kowiki 문서·한국어 라벨 보유(로컬 색인에서 검색 가능). 순환 금지.
출력 행: {"chain_id", "seed": {qid,label}, "hops": [{"from", "prop", "prop_label", "to", "to_label"}], "answer": label, "answer_aliases": [...], "n_hops"}
질문 자연어화는 wd_question.py(교사) 가 맡는다. Wikidata API 는 배치 50·User-Agent 명시·캐시(out/wd_cache.jsonl).
사용: python3 wd_chain.py --titles-from out/corpus/passages.jsonl --n-chains 10000 --out out/d1/chains.jsonl [--min-passages 6] [--rps 3]
"""
import argparse, json, os, random, time, urllib.parse, urllib.request, urllib.error, collections, threading

API = "https://www.wikidata.org/w/api.php"; UA = "alpha-sdg-search/0.1 (research; contact: cjaidivision@gmail.com)"
PROPS = {  # 의미 있는 관계(값이 개체). 너무 일반적인 P31/P279/P106/P27 은 제외
 "P17": "국가", "P131": "행정구역", "P19": "출생지", "P20": "사망지", "P69": "출신 학교", "P108": "소속 기관(고용주)", "P50": "저자", "P57": "감독", "P58": "각본가", "P86": "작곡가",
 "P175": "가수/연주자", "P264": "음반사", "P176": "제조사", "P112": "설립자", "P127": "소유자", "P749": "상위 조직", "P159": "본사 소재지", "P36": "수도", "P35": "국가 원수", "P6": "정부 수반",
 "P26": "배우자", "P22": "아버지", "P25": "어머니", "P40": "자녀", "P3373": "형제자매", "P169": "최고경영자", "P54": "소속 팀", "P118": "리그", "P115": "홈 경기장", "P138": "이름의 유래",
 "P84": "건축가", "P170": "창작자", "P123": "출판사", "P1376": "수도인 곳", "P206": "인접 수역", "P37": "공용어", "P38": "통화", "P361": "상위 단위", "P710": "참가자", "P800": "대표작",
 "P39": "직위", "P463": "소속 단체", "P102": "소속 정당", "P1416": "소속", "P737": "영향을 받은 대상", "P144": "원작", "P179": "시리즈", "P155": "이전 작품/항목", "P156": "다음 작품/항목",
 "P1037": "책임자(관리자)", "P488": "의장/대표", "P1344": "참가한 대회", "P137": "운영자", "P466": "입주자", "P276": "위치", "P740": "결성 장소", "P1830": "소유물",
 "P2388": "직위 담당", "P1435": "문화유산 지정", "P97": "귀족 작위", "P611": "종교 수도회", "P1066": "스승", "P802": "제자", "P1327": "협력자", 
}
_lock = threading.Lock(); _last = [0.0]
def api(params, rps):
    with _lock:                       # 토큰버킷: 다음 슬롯을 예약만 하고 sleep 은 락 밖에서(병렬 in-flight 허용)
        slot = max(_last[0] + 1.0 / rps, time.time()); _last[0] = slot
    time.sleep(max(0.0, slot - time.time()))
    q = urllib.parse.urlencode({**params, "format": "json"}); req = urllib.request.Request(API + "?" + q, headers={"User-Agent": UA})
    for i in range(7):
        try: return json.load(urllib.request.urlopen(req, timeout=60))
        except urllib.error.HTTPError as e:
            err = e; time.sleep(15 if e.code == 429 else 3 * (i + 1))     # 429: 15초 백오프
        except Exception as e:
            time.sleep(3 * (i + 1)); err = e
    raise err

class Store:
    def __init__(self, path, rps):
        self.path, self.rps, self.ent = path, rps, {}
        if os.path.exists(path):
            for l in open(path):
                d = json.loads(l); self.ent[d["id"]] = d
        self.f = open(path, "a")
    def _save(self, e):
        self.ent[e["id"]] = e; self.f.write(json.dumps(e, ensure_ascii=False) + "\n"); self.f.flush()
    def slim(self, raw):
        claims = {}
        for p, cl in (raw.get("claims") or {}).items():
            if p not in PROPS: continue
            vals = []
            for c in cl:
                if c.get("rank") == "deprecated": continue
                dv = ((c.get("mainsnak") or {}).get("datavalue") or {}).get("value")
                if isinstance(dv, dict) and dv.get("entity-type") == "item": vals.append(dv["id"])
            if vals: claims[p] = vals
        return {"id": raw["id"], "label": ((raw.get("labels") or {}).get("ko") or {}).get("value"), "aliases": [a["value"] for a in ((raw.get("aliases") or {}).get("ko") or [])],
                "kowiki": ((raw.get("sitelinks") or {}).get("kowiki") or {}).get("title"), "claims": claims}
    def get_many(self, ids):
        need = [i for i in ids if i not in self.ent]
        for i in range(0, len(need), 50):
            batch = need[i:i + 50]
            d = api({"action": "wbgetentities", "ids": "|".join(batch), "props": "labels|aliases|claims|sitelinks", "languages": "ko", "sitefilter": "kowiki"}, self.rps)
            for qid, raw in (d.get("entities") or {}).items():
                if "missing" in raw: self._save({"id": qid, "missing": True}); continue
                self._save(self.slim(raw))
        return {i: self.ent.get(i) for i in ids}
    def by_titles(self, titles):
        out = {}
        for i in range(0, len(titles), 50):
            batch = titles[i:i + 50]
            d = api({"action": "wbgetentities", "sites": "kowiki", "titles": "|".join(batch), "props": "labels|aliases|claims|sitelinks", "languages": "ko", "sitefilter": "kowiki"}, self.rps)
            for qid, raw in (d.get("entities") or {}).items():
                if "missing" in raw or not qid.startswith("Q"): continue
                e = self.slim(raw); self._save(e); out[e["kowiki"]] = e
        return out

GENERIC = {"P17", "P37", "P38", "P30", "P36", "P1376", "P361", "P131"}   # 국가·공용어·통화·대륙·수도·상위단위·행정구역: 정답이 뻔해짐 → 마지막 홉 금지, 연쇄당 ≤1
def walk(store, seed, rng, min_hops, max_hops):
    """홉마다 단일값 후보 전부를 한 번에 배치 조회(호출 1회) → 유효 후보 중 무작위 선택. 호출 수 ≈ 홉 수."""
    cur, chain, seen = seed, [], {seed["id"]}
    target = rng.randint(min_hops, max_hops)
    for h in range(target):
        n_generic = sum(1 for c in chain if c["prop"] in GENERIC); last = (h == target - 1)
        cands = [(p, v[0]) for p, v in cur["claims"].items() if p in PROPS and len(v) == 1 and v[0] not in seen and not (p in GENERIC and (last or n_generic >= 1))]
        if not cands: break
        rng.shuffle(cands); cands = cands[:20]
        ents = store.get_many([q for _, q in cands])
        good = [(p, ents[q]) for p, q in cands if ents.get(q) and not ents[q].get("missing") and ents[q].get("label") and ents[q].get("kowiki") and (ents[q]["claims"] or h == target - 1)]
        if not good: break
        p, e = good[0]; chain.append({"from": cur["id"], "from_label": cur["label"], "prop": p, "prop_label": PROPS[p], "to": e["id"], "to_label": e["label"]}); seen.add(e["id"]); cur = e
    return chain

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--titles-from", required=True); ap.add_argument("--n-chains", type=int, default=10000); ap.add_argument("--out", required=True)
    ap.add_argument("--min-passages", type=int, default=6); ap.add_argument("--min-hops", type=int, default=3); ap.add_argument("--max-hops", type=int, default=6); ap.add_argument("--rps", type=float, default=10.0); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(); rng = random.Random(a.seed); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    cnt = collections.Counter()
    for l in open(a.titles_from):
        d = json.loads(l)
        if d.get("source") == "kowiki": cnt[d["title"]] += 1
    titles = [t for t, c in cnt.items() if c >= a.min_passages and not any(x in t for x in ("목록", "년", "월 ", "일 ", "(동음이의)"))]
    rng.shuffle(titles); print(f"candidate titles {len(titles)} (of {len(cnt)})", flush=True)
    store = Store(os.path.join(os.path.dirname(a.out) or ".", "wd_cache.jsonl"), a.rps)
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out): done.add(json.loads(l)["seed"]["qid"])
    out = open(a.out, "a"); n = len(done); hist = collections.Counter(); t0 = time.time(); ti = 0
    from concurrent.futures import ThreadPoolExecutor, as_completed
    olock = threading.Lock()
    def build(e):
        r = random.Random(hash(e["id"]) & 0xffffffff)
        for attempt in range(2):
            chain = walk(store, e, r, a.min_hops, a.max_hops)
            if len(chain) >= a.min_hops: return e, chain
        return e, chain
    with ThreadPoolExecutor(a.workers) as ex:
        while n < a.n_chains and ti < len(titles):
            batch = titles[ti:ti + 200]; ti += 200
            ents = [e for e in store.by_titles(batch).values() if e["id"] not in done and e.get("label") and e["claims"]]
            for f in as_completed([ex.submit(build, e) for e in ents]):
                e, chain = f.result()
                if len(chain) < a.min_hops: continue
                last = store.ent[chain[-1]["to"]]
                with olock:
                    if e["id"] in done or n >= a.n_chains: continue
                    out.write(json.dumps({"chain_id": f"wd:{e['id']}:{len(chain)}:{rng.randrange(1<<20):05x}", "seed": {"qid": e["id"], "label": e["label"], "kowiki": e["kowiki"]}, "hops": chain,
                                          "answer": last["label"], "answer_aliases": last.get("aliases", []), "answer_kowiki": last.get("kowiki"), "n_hops": len(chain)}, ensure_ascii=False) + "\n"); out.flush()
                    done.add(e["id"]); n += 1; hist[len(chain)] += 1
                    if n % 200 == 0: print(f"[{n}/{a.n_chains}] {time.time()-t0:.0f}s hops={dict(sorted(hist.items()))} titles_used={ti} cache={len(store.ent)}", flush=True)
    print("DONE", n, dict(sorted(hist.items())), flush=True)

if __name__ == "__main__":
    main()

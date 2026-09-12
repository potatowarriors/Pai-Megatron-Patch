#!/usr/bin/env python3
"""build_index.py — passages.jsonl → bm25s 색인 (한국어: 어절 + 한글 2-gram, 영문/숫자: 소문자 토큰). 결정적·무제한 로컬 검색(유료 API 폐기 결정 2026-09-12).
사용: python3 build_index.py --passages out/corpus/passages.jsonl --out out/index/bm25  → out/index/bm25/ (bm25s 저장) + docs.jsonl(순서 정렬 메타)
"""
import argparse, json, os, re, time
import bm25s

TOK = re.compile(r"[가-힣]+|[a-z0-9]+(?:\.[a-z0-9]+)*|[一-鿿]+|[ぁ-んァ-ン]+")
def tokenize(text):
    out = []
    for m in TOK.finditer(text.lower()):
        w = m.group(0)
        if re.match(r"[가-힣]", w):
            out.append(w)
            if len(w) >= 2: out.extend(w[i:i + 2] for i in range(len(w) - 1))
        else: out.append(w)
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--passages", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); t0 = time.time(); metas = []; corpus_tokens = []
    with open(a.passages) as f, open(os.path.join(a.out, "docs.jsonl"), "w") as m:
        for i, l in enumerate(f):
            d = json.loads(l); corpus_tokens.append(tokenize(d["text"]))
            m.write(json.dumps({"doc_id": d["doc_id"], "title": d["title"], "url": d["url"], "source": d["source"], "text": d["text"]}, ensure_ascii=False) + "\n")
            if i % 500000 == 0 and i: print(f"tokenized {i} ({time.time()-t0:.0f}s)", flush=True)
    print(f"tokenize done n={len(corpus_tokens)} ({time.time()-t0:.0f}s)", flush=True)
    retriever = bm25s.BM25(k1=1.2, b=0.75)
    retriever.index(corpus_tokens)
    retriever.save(a.out)
    print(f"index saved {a.out} ({time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__":
    main()

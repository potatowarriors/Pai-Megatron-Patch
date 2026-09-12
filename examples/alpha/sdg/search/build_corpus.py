#!/usr/bin/env python3
"""build_corpus.py — 트랙 D 로컬 검색 코퍼스: 한국어 위키백과 덤프 + NIKL 신문 기사 → passage jsonl (2026-09-12, 유료 검색 API 폐기 결정).

출력 행: {"doc_id", "title", "url", "source": "kowiki|nikl_news", "text"}  — passage ≈ 최대 700자, 문단 경계 우선.
위키 마크업은 경량 정리(템플릿·표·ref·파일 제거, 링크 텍스트 보존). 넘겨주기·문서이름공간≠0 제외. 본문 200자 미만 문서 제외.
사용: python3 build_corpus.py --kowiki /home/work/vidsearch/datasets_raw/kowiki/kowiki-latest-pages-articles.xml.bz2 \
        --nikl "/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/datasets/raw/Korea_NewsPaper_Corpus/NIKL_NEWSPAPER_2025_v1.0/*.json" --out out/corpus/passages.jsonl
"""
import argparse, bz2, json, re, glob, os, html, hashlib
import xml.etree.ElementTree as ET

def strip_templates(t):
    # 중첩 {{ }} 제거
    out, depth, i = [], 0, 0
    while i < len(t):
        if t.startswith("{{", i): depth += 1; i += 2; continue
        if t.startswith("}}", i) and depth: depth -= 1; i += 2; continue
        if not depth: out.append(t[i])
        i += 1
    return "".join(out)

def strip_tables(t):
    return re.sub(r"\{\|.*?\|\}", " ", t, flags=re.S)

def clean_wiki(t):
    t = re.sub(r"<!--.*?-->", "", t, flags=re.S)
    t = re.sub(r"<ref[^>]*/>", "", t); t = re.sub(r"<ref[^>]*>.*?</ref>", "", t, flags=re.S)
    t = strip_templates(strip_tables(t))
    t = re.sub(r"\[\[(?:파일|File|그림|Image|분류|Category):[^\]]*\]\]", "", t, flags=re.I)
    t = re.sub(r"\[\[([^\]|]*)\|([^\]]*)\]\]", r"\2", t); t = re.sub(r"\[\[([^\]]*)\]\]", r"\1", t)
    t = re.sub(r"\[https?://[^\s\]]+\s*([^\]]*)\]", r"\1", t)
    t = re.sub(r"<[^>]+>", "", t); t = html.unescape(t)
    t = re.sub(r"'{2,}", "", t)
    t = re.sub(r"^\s*={2,}\s*(.*?)\s*={2,}\s*$", r"\n\1\n", t, flags=re.M)     # 제목은 줄로
    t = re.sub(r"^[*#:;]+\s*", "", t, flags=re.M)
    t = re.sub(r"[ \t]+", " ", t); t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

STOP_SECTIONS = ("같이 보기", "각주", "외부 링크", "참고 문헌", "참고 자료", "주석", "출처")
def passages(text, maxlen=700):
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out, cur = [], ""
    for p in paras:
        if p.strip() in STOP_SECTIONS: break
        if len(p) < 15: continue
        if len(cur) + len(p) + 1 <= maxlen: cur = (cur + "\n" + p).strip()
        else:
            if cur: out.append(cur)
            while len(p) > maxlen: out.append(p[:maxlen]); p = p[maxlen:]
            cur = p
    if cur: out.append(cur)
    return [x for x in out if len(x) >= 60]

def iter_kowiki(path):
    ns = "{http://www.mediawiki.org/xml/export-0.11/}"
    with bz2.open(path, "rb") as f:
        for ev, el in ET.iterparse(f, events=("end",)):
            if not el.tag.endswith("}page"): continue
            nsid = el.findtext(f"{ns}ns") or el.findtext("ns")
            title = el.findtext(f"{ns}title") or el.findtext("title") or ""
            red = el.find(f"{ns}redirect") if el.find(f"{ns}redirect") is not None else el.find("redirect")
            rev = el.find(f"{ns}revision") if el.find(f"{ns}revision") is not None else el.find("revision")
            text = (rev.findtext(f"{ns}text") if rev is not None else None) or (rev.findtext("text") if rev is not None else None) or ""
            el.clear()
            if nsid not in ("0", None) or red is not None or not text: continue
            if title.startswith(("위키백과:", "틀:", "분류:", "파일:", "포털:")): continue
            body = clean_wiki(text)
            if len(body) < 200: continue
            yield title, body

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--kowiki"); ap.add_argument("--nikl"); ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    n_doc = n_pas = 0
    with open(a.out, "w") as o:
        if a.kowiki:
            for title, body in iter_kowiki(a.kowiki):
                did = "w" + hashlib.sha1(title.encode()).hexdigest()[:12]
                for i, p in enumerate(passages(body)):
                    o.write(json.dumps({"doc_id": f"{did}#{i}", "title": title, "url": "https://ko.wikipedia.org/wiki/" + title.replace(" ", "_"), "source": "kowiki", "text": (title + "\n" + p) if i == 0 else p}, ensure_ascii=False) + "\n"); n_pas += 1
                n_doc += 1
                if n_doc % 50000 == 0: print(f"[kowiki] docs {n_doc} passages {n_pas}", flush=True)
                if a.limit and n_doc >= a.limit: break
            print(f"[kowiki] DONE docs {n_doc} passages {n_pas}", flush=True)
        if a.nikl:
            nd = npz = 0
            for fp in sorted(glob.glob(a.nikl)):
                try: d = json.load(open(fp, encoding="utf-8"))
                except Exception as e: print("skip", fp, e); continue
                for doc in d.get("document", []):
                    md = doc.get("metadata") or {}; title = md.get("title") or ""; body = "\n\n".join(re.sub(r"</?p>", "", (x.get("form") or "")).strip() for x in doc.get("paragraph", []) if x.get("form"))
                    if len(body) < 200: continue
                    did = "n" + hashlib.sha1((doc.get("id") or title).encode()).hexdigest()[:12]
                    for i, p in enumerate(passages(body)):
                        o.write(json.dumps({"doc_id": f"{did}#{i}", "title": title, "url": f"nikl://{md.get('publisher','')}/{md.get('date','')}/{doc.get('id','')}", "source": "nikl_news", "date": md.get("date"), "publisher": md.get("publisher"), "text": (title + "\n" + p) if i == 0 else p}, ensure_ascii=False) + "\n"); npz += 1
                    nd += 1
                print(f"[nikl] {os.path.basename(fp)} docs {nd} passages {npz}", flush=True)
            print(f"[nikl] DONE docs {nd} passages {npz}", flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()

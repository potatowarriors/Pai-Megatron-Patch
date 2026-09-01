#!/usr/bin/env python3
# Copyright (c) 2026 alpha team. Apache-2.0.
"""reasoning_content 에 새어 든 교사 프롬프트 스캐폴딩 용어를 언어별 자연어로 치환 (KNOWN_ISSUES 2026-09-01 ⑤).

대상(reasoning 만, content 불변): `<identity-facts>`/identity-facts/identity facts, 사실 블록 섹션명(SCALE·ARCHITECTURE·ORIGIN·
DEVELOPED BY·NOT DISCLOSED·UNDISCLOSED), 시드 enum(lead_only/all_members). 원본은 data_backup_scrub_prev/ 에 보존.
사용: python3 scrub_reasoning_scaffold.py --dataset-dir /…/alpha-SFT-Identity-v2 [--dry-run]
"""
import argparse, json, re, shutil, sys
from collections import Counter
from pathlib import Path

IDF = {"ko": "제 정체성 정보", "en": "my identity information", "zh": "我的身份信息", "ja": "私のアイデンティティ情報",
       "es": "mi información de identidad", "de": "meine Identitätsinformationen", "fr": "mes informations d'identité",
       "it": "le mie informazioni di identità", "pt": "minhas informações de identidade"}
SEC_KO = {"SCALE": "규모", "ARCHITECTURE": "아키텍처", "ORIGIN": "출처", "DEVELOPED BY": "개발 주체", "NOT DISCLOSED": "비공개", "UNDISCLOSED": "비공개", "NAME": "명칭"}
SEC_EN = {"SCALE": "scale", "ARCHITECTURE": "architecture", "ORIGIN": "origin", "DEVELOPED BY": "developer", "NOT DISCLOSED": "not-disclosed", "UNDISCLOSED": "undisclosed", "NAME": "name"}
SEC_JA = {"SCALE": "規模", "ARCHITECTURE": "アーキテクチャ", "ORIGIN": "出自", "DEVELOPED BY": "開発主体", "NOT DISCLOSED": "非公開", "UNDISCLOSED": "非公開", "NAME": "名称"}
SEC_ZH = {"SCALE": "规模", "ARCHITECTURE": "架构", "ORIGIN": "来源", "DEVELOPED BY": "开发主体", "NOT DISCLOSED": "未公开", "UNDISCLOSED": "未公开", "NAME": "名称"}
ENUM_KO = {"lead_only": "리드만 명시하는", "all_members": "전원을 명시하는"}
ENUM_EN = {"lead_only": "lead-only", "all_members": "all-members"}
TOK = re.compile(r"<?/?identity[- _]facts>?", re.I)
RESID = re.compile(r"identity[- _]facts|<identity|(?<![A-Za-z])(SCALE|ARCHITECTURE|ORIGIN|UNDISCLOSED|DEVELOPED BY|NOT DISCLOSED)(?![A-Za-z])|lead_only|all_members")  # CJK 인접 허용


def scrub(text: str, lang: str) -> str:
    t = text
    idf = IDF.get(lang, IDF["en"])
    # 관사·수식 정리 후 토큰 치환
    t = re.sub(r"(the|The)\s+identity[- _]facts", idf, t)
    t = re.sub(r"제공된\s+<?identity[- _]facts>?", idf, t)
    t = TOK.sub(idf, t)
    # 섹션명
    sec = {"ko": SEC_KO, "ja": SEC_JA, "zh": SEC_ZH}.get(lang, SEC_EN)
    for k, v in sec.items():
        if lang == "ko":
            t = re.sub(rf"'?{re.escape(k)}'?\s*(항목|섹션|정보|section|entry|field)?", lambda m, v=v: v + (" " + m.group(1) if m.group(1) else ""), t)
        else:
            # CJK 인접 시 \b 가 성립하지 않으므로 대문자 토큰은 경계 없이 치환 (섹션명은 항상 대문자 고유 표기)
            t = re.sub(rf"'?(?<![A-Za-z]){re.escape(k)}(?![A-Za-z])'?", v, t)
    # 시드 enum
    en = ENUM_KO if lang == "ko" else ENUM_EN
    for k, v in en.items():
        t = re.sub(rf"{k}\s*(형식|form|format)?", lambda m, v=v: v + (" 형식" if lang == "ko" else " form"), t)
    t = re.sub(r"  +", " ", t)
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    data = a.dataset_dir / "data"
    if not a.dry_run:
        bk = a.dataset_dir / "data_backup_scrub_prev"
        if not bk.exists():
            shutil.copytree(data, bk, ignore=shutil.ignore_patterns("train_x*.jsonl"))
            print("backup →", bk)
    tot = Counter()
    examples = []
    for fn in ("train.jsonl", "eval.jsonl"):
        p = data / fn
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        changed = 0
        for r in rows:
            lang = (r.get("metadata") or {}).get("language", "en")
            for m in r["messages"]:
                if m["role"] != "assistant":
                    continue
                rc = m.get("reasoning_content")
                if not rc or not RESID.search(rc):
                    continue
                new = scrub(rc, lang)
                if RESID.search(new):
                    tot["residual"] += 1
                if new != rc:
                    changed += 1
                    if len(examples) < 4:
                        examples.append((lang, rc[:150], new[:150]))
                    m["reasoning_content"] = new
        tot[fn] = changed
        if not a.dry_run:
            with open(p, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("치환 턴 수:", {k: v for k, v in tot.items() if k != "residual"}, "| 잔여:", tot["residual"])
    for lang, b, c in examples:
        print(f"[{lang}] BEFORE: {b}\n     AFTER : {c}")
    return 1 if tot["residual"] else 0


if __name__ == "__main__":
    sys.exit(main())

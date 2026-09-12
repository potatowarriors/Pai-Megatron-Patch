#!/usr/bin/env python3
"""u_templates.py — 트랙 U(구조화 출력·사용성) 제약 템플릿 + 프로그램 검증기 (2026-09-12, 사용자 승인 계획).

각 템플릿 = make(seed_prompt, lang, rng) -> dict(user=<프롬프트+제약 지시>, validate=fn(content)->(ok, why), meta={...}, kind=...)
  kind: "single" (단일턴) | "clarify" | "refuse" | "followup" (2턴: 1턴 답변 생성 후 변환 지시)
검증은 전부 결정적(프로그램). 어투·거절 품질처럼 프로그램으로 못 재는 항목은 최소 휴리스틱 + 교차 심판(u_generate) 이 맡는다.
EN/KO 지시문을 같은 검증기로 검사한다. 한국어 길이 제약은 글자수(자), 영어는 단어수.
"""
import json, random, re, csv, io
import yaml, jsonschema

H = re.compile(r"[가-힣]"); L = re.compile(r"[A-Za-z]")
def lang_ratio(s):
    h, l = len(H.findall(s)), len(L.findall(s)); t = h + l
    return (h / t if t else 0.0, l / t if t else 0.0)
def words(s): return [w for w in re.split(r"\s+", s.strip()) if w]
def sentences(s):
    parts = re.split(r"(?<=[.!?。！？])\s+|(?<=다\.)\s+|\n+", s.strip())
    return [p for p in parts if p.strip()]
def paragraphs(s): return [p for p in re.split(r"\n\s*\n", s.strip()) if p.strip()]
def strip_fence(s, lang_tag=None):
    m = re.fullmatch(r"\s*```(?:[a-zA-Z]+)?\s*\n(.*?)\n\s*```\s*", s, flags=re.S)
    return (m.group(1), True) if m else (s, False)

# ───────────────────────── JSON 스키마 라이브러리 (주제 무관 12종) ─────────────────────────
SCHEMAS = {
 "summary": {"type": "object", "required": ["title", "key_points", "one_line_summary"], "additionalProperties": False,
             "properties": {"title": {"type": "string"}, "key_points": {"type": "array", "minItems": 3, "maxItems": 5, "items": {"type": "string"}}, "one_line_summary": {"type": "string"}}},
 "qa": {"type": "object", "required": ["question", "answer", "confidence", "assumptions"], "additionalProperties": False,
        "properties": {"question": {"type": "string"}, "answer": {"type": "string"}, "confidence": {"type": "number", "minimum": 0, "maximum": 1}, "assumptions": {"type": "array", "items": {"type": "string"}}}},
 "plan": {"type": "object", "required": ["goal", "steps"], "additionalProperties": False,
          "properties": {"goal": {"type": "string"}, "steps": {"type": "array", "minItems": 3, "items": {"type": "object", "required": ["step", "action", "rationale"], "additionalProperties": False,
                         "properties": {"step": {"type": "integer"}, "action": {"type": "string"}, "rationale": {"type": "string"}}}}}},
 "entities": {"type": "object", "required": ["people", "organizations", "locations", "dates", "topics"], "additionalProperties": False,
              "properties": {k: {"type": "array", "items": {"type": "string"}} for k in ("people", "organizations", "locations", "dates", "topics")}},
 "classification": {"type": "object", "required": ["category", "confidence", "reasoning"], "additionalProperties": False,
                    "properties": {"category": {"type": "string", "enum": ["question", "task_request", "creative", "advice", "complaint", "other"]}, "confidence": {"type": "number", "minimum": 0, "maximum": 1}, "reasoning": {"type": "string"}}},
 "pros_cons": {"type": "object", "required": ["subject", "pros", "cons", "verdict"], "additionalProperties": False,
               "properties": {"subject": {"type": "string"}, "pros": {"type": "array", "minItems": 2, "items": {"type": "string"}}, "cons": {"type": "array", "minItems": 2, "items": {"type": "string"}}, "verdict": {"type": "string"}}},
 "checklist": {"type": "object", "required": ["title", "items"], "additionalProperties": False,
               "properties": {"title": {"type": "string"}, "items": {"type": "array", "minItems": 3, "items": {"type": "object", "required": ["item", "priority"], "additionalProperties": False,
                              "properties": {"item": {"type": "string"}, "priority": {"type": "string", "enum": ["high", "medium", "low"]}}}}}},
 "comparison": {"type": "object", "required": ["items", "recommendation"], "additionalProperties": False,
                "properties": {"items": {"type": "array", "minItems": 2, "items": {"type": "object", "required": ["name", "strengths", "weaknesses", "score"], "additionalProperties": False,
                               "properties": {"name": {"type": "string"}, "strengths": {"type": "array", "items": {"type": "string"}}, "weaknesses": {"type": "array", "items": {"type": "string"}}, "score": {"type": "integer", "minimum": 1, "maximum": 10}}}},
                               "recommendation": {"type": "string"}}},
 "faq": {"type": "object", "required": ["faqs"], "additionalProperties": False,
         "properties": {"faqs": {"type": "array", "minItems": 3, "maxItems": 6, "items": {"type": "object", "required": ["q", "a"], "additionalProperties": False, "properties": {"q": {"type": "string"}, "a": {"type": "string"}}}}}},
 "rewrite": {"type": "object", "required": ["original_intent", "response", "notes"], "additionalProperties": False,
             "properties": {"original_intent": {"type": "string"}, "response": {"type": "string"}, "notes": {"type": "array", "items": {"type": "string"}}}},
 "risk": {"type": "object", "required": ["risks"], "additionalProperties": False,
          "properties": {"risks": {"type": "array", "minItems": 2, "items": {"type": "object", "required": ["risk", "likelihood", "mitigation"], "additionalProperties": False,
                         "properties": {"risk": {"type": "string"}, "likelihood": {"type": "string", "enum": ["low", "medium", "high"]}, "mitigation": {"type": "string"}}}}}},
 "answer_meta": {"type": "object", "required": ["answer", "tone", "language", "follow_up_questions"], "additionalProperties": False,
                 "properties": {"answer": {"type": "string"}, "tone": {"type": "string", "enum": ["formal", "casual", "neutral"]}, "language": {"type": "string"}, "follow_up_questions": {"type": "array", "maxItems": 3, "items": {"type": "string"}}}},
}

def _json_of(content, allow_fence):
    body, fenced = strip_fence(content)
    if fenced and not allow_fence: return None, "fence_not_allowed"
    try: return json.loads(body), None
    except Exception:
        return None, "not_json"

def t_json_schema(prompt, lang, rng):
    name = rng.choice(list(SCHEMAS)); schema = SCHEMAS[name]; allow_fence = rng.random() < 0.4
    if lang == "ko":
        instr = ("\n\n위 요청에 대한 답을 아래 JSON 스키마를 만족하는 JSON 객체 **하나로만** 출력하세요. " + ("```json 코드펜스 하나는 허용하지만 " if allow_fence else "코드펜스·설명·머리말 없이 ") + "JSON 외의 텍스트는 쓰지 마세요.\n스키마:\n" + json.dumps(schema, ensure_ascii=False))
    else:
        instr = ("\n\nRespond to the request above ONLY with a single JSON object that conforms to the JSON Schema below. " + ("A single ```json code fence is allowed, but " if allow_fence else "No code fences, no prose, no preamble: ") + "nothing outside the JSON.\nSchema:\n" + json.dumps(schema))
    def validate(c):
        obj, why = _json_of(c, allow_fence)
        if why: return False, why
        try: jsonschema.validate(obj, schema)
        except jsonschema.ValidationError as e: return False, "schema:" + e.message[:80]
        return True, None
    return {"user": prompt + instr, "validate": validate, "meta": {"family": "json_schema", "schema": name, "allow_fence": allow_fence}, "kind": "single"}

def t_json_keys(prompt, lang, rng):
    keys = rng.sample(["summary", "answer", "steps", "key_terms", "caveats", "next_actions", "examples", "sources_needed", "estimated_time", "difficulty"], rng.randint(3, 5))
    if lang == "ko": instr = f"\n\n답변을 키가 정확히 {', '.join(keys)} 인 JSON 객체 하나로만 출력하세요(다른 키 금지, JSON 밖 텍스트 금지, 코드펜스 금지)."
    else: instr = f"\n\nReturn ONLY a JSON object whose keys are exactly: {', '.join(keys)}. No other keys, no text outside the JSON, no code fences."
    def validate(c):
        obj, why = _json_of(c, False)
        if why: return False, why
        if not isinstance(obj, dict): return False, "not_object"
        if set(obj) != set(keys): return False, f"keys:{sorted(set(obj) ^ set(keys))[:3]}"
        return True, None
    return {"user": prompt + instr, "validate": validate, "meta": {"family": "json_keys", "keys": keys}, "kind": "single"}

def t_csv(prompt, lang, rng):
    n = rng.randint(3, 7); cols = rng.choice([["item", "description", "priority"], ["step", "action", "owner", "duration"], ["option", "pros", "cons", "score"], ["term", "definition", "example"], ["question", "answer", "confidence"]])
    if lang == "ko": instr = f"\n\n답변을 CSV 로만 출력하세요. 첫 줄은 헤더 `{','.join(cols)}` 이고 데이터 행은 정확히 {n}개, 그 외 텍스트·코드펜스는 쓰지 마세요. 쉼표가 들어가는 값은 큰따옴표로 감싸세요."
    else: instr = f"\n\nAnswer as CSV only: header row `{','.join(cols)}`, then exactly {n} data rows. No other text, no code fences. Quote any value containing a comma."
    def validate(c):
        body, fenced = strip_fence(c)
        if fenced: return False, "fence_not_allowed"
        try: rows = list(csv.reader(io.StringIO(body.strip())))
        except Exception: return False, "csv_parse"
        rows = [r for r in rows if any(x.strip() for x in r)]
        if not rows or [x.strip() for x in rows[0]] != cols: return False, "header"
        data = rows[1:]
        if len(data) != n: return False, f"rows:{len(data)}"
        if any(len(r) != len(cols) for r in data): return False, "ragged"
        return True, None
    return {"user": prompt + instr, "validate": validate, "meta": {"family": "csv", "cols": cols, "n": n}, "kind": "single"}

def t_md_table(prompt, lang, rng):
    n = rng.randint(3, 6); cols = rng.choice([["항목", "설명", "비고"] if lang == "ko" else ["Item", "Description", "Notes"], ["단계", "할 일", "예상 시간"] if lang == "ko" else ["Step", "Action", "Time"], ["선택지", "장점", "단점"] if lang == "ko" else ["Option", "Pros", "Cons"]])
    if lang == "ko": instr = f"\n\n답변 전체를 마크다운 표 하나로만 작성하세요. 열은 정확히 `{' | '.join(cols)}` 이고 데이터 행은 정확히 {n}개입니다. 표 앞뒤에 다른 문장을 쓰지 마세요."
    else: instr = f"\n\nWrite the entire answer as a single Markdown table with exactly these columns: `{' | '.join(cols)}` and exactly {n} data rows. No sentences before or after the table."
    def validate(c):
        lines = [l for l in c.strip().splitlines() if l.strip()]
        if len(lines) < 2 or not all(l.strip().startswith("|") for l in lines): return False, "not_only_table"
        hdr = [x.strip() for x in lines[0].strip().strip("|").split("|")]
        if hdr != cols: return False, "header"
        if not re.fullmatch(r"\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?", lines[1].strip()): return False, "separator"
        data = lines[2:]
        if len(data) != n: return False, f"rows:{len(data)}"
        if any(len([x for x in l.strip().strip("|").split("|")]) != len(cols) for l in data): return False, "ragged"
        return True, None
    return {"user": prompt + instr, "validate": validate, "meta": {"family": "md_table", "cols": cols, "n": n}, "kind": "single"}

def t_yaml(prompt, lang, rng):
    keys = rng.sample(["summary", "details", "recommendations", "warnings", "references", "estimated_effort"], rng.randint(3, 4))
    if lang == "ko": instr = f"\n\n답변을 YAML 로만 출력하세요. 최상위 키는 정확히 {', '.join(keys)} 이며 코드펜스와 다른 텍스트는 쓰지 마세요."
    else: instr = f"\n\nAnswer in YAML only, with exactly these top-level keys: {', '.join(keys)}. No code fences and no other text."
    def validate(c):
        body, fenced = strip_fence(c)
        if fenced: return False, "fence_not_allowed"
        try: obj = yaml.safe_load(body)
        except Exception: return False, "yaml_parse"
        if not isinstance(obj, dict): return False, "not_mapping"
        if set(obj) != set(keys): return False, f"keys:{sorted(set(obj) ^ set(keys))[:3]}"
        return True, None
    return {"user": prompt + instr, "validate": validate, "meta": {"family": "yaml", "keys": keys}, "kind": "single"}

def t_regex(prompt, lang, rng):
    kind = rng.choice(["answer_tag", "single_line", "number_only", "bullets_exact", "numbered_exact"])
    n = rng.randint(3, 6)
    if kind == "answer_tag":
        instr = "\n\n최종 답을 <answer></answer> 태그 안에 넣고, 태그 밖에는 아무것도 쓰지 마세요." if lang == "ko" else "\n\nPut your final answer inside <answer></answer> tags and write nothing outside the tags."
        v = lambda c: (bool(re.fullmatch(r"\s*<answer>.*?</answer>\s*", c, flags=re.S)) and len(re.findall(r"<answer>", c)) == 1, "tag")
    elif kind == "single_line":
        instr = "\n\n답변은 줄바꿈 없이 한 줄로만 쓰세요." if lang == "ko" else "\n\nAnswer in a single line with no line breaks."
        v = lambda c: ("\n" not in c.strip() and len(c.strip()) > 0, "multiline")
    elif kind == "number_only":
        instr = "\n\n답을 숫자 하나로만 출력하세요(단위·설명·기호 없이). 정확한 값을 모르면 가장 합리적인 추정치 하나를 숫자로만 쓰세요." if lang == "ko" else "\n\nOutput a single number only (no units, no words, no symbols). If the exact value is unknown, give your best single estimate as a bare number."
        v = lambda c: (bool(re.fullmatch(r"\s*-?\d+(?:[.,]\d+)?\s*", c)), "not_number")
    elif kind == "bullets_exact":
        instr = f"\n\n답변을 글머리표(`- `로 시작) 정확히 {n}개로만 구성하세요. 글머리표 외의 줄은 쓰지 마세요." if lang == "ko" else f"\n\nAnswer with exactly {n} bullet points (lines starting with `- `) and nothing else."
        v = lambda c: (all(l.strip().startswith("- ") for l in c.strip().splitlines() if l.strip()) and sum(1 for l in c.strip().splitlines() if l.strip().startswith("- ")) == n, "bullets")
    else:
        instr = f"\n\n답변을 `1.`~`{n}.` 번호 목록 정확히 {n}개 항목으로만 쓰세요. 그 외의 줄은 쓰지 마세요." if lang == "ko" else f"\n\nAnswer as a numbered list with exactly {n} items (`1.` to `{n}.`) and no other lines."
        v = lambda c: ([re.match(r"\s*(\d+)\.", l).group(1) for l in c.strip().splitlines() if l.strip() and re.match(r"\s*\d+\.", l)] == [str(i) for i in range(1, n + 1)] and all(re.match(r"\s*\d+\.", l) for l in c.strip().splitlines() if l.strip()), "numbered")
    def validate(c):
        ok, why = v(c); return (True, None) if ok else (False, why)
    return {"user": prompt + instr, "validate": validate, "meta": {"family": "regex", "kind": kind, "n": n}, "kind": "single"}

def t_toolcall_json(prompt, lang, rng):
    tools = rng.choice([("send_email", ["to", "subject", "body"]), ("create_task", ["title", "due_date", "priority"]), ("search_docs", ["query", "max_results"]), ("schedule_meeting", ["title", "participants", "start_time", "duration_minutes"]), ("translate_text", ["text", "target_language"])])
    name, args = tools
    if lang == "ko": instr = f"\n\n이 요청을 처리하기 위해 도구 `{name}` 을 호출한다고 가정하고, 답변으로 JSON 객체 하나만 출력하세요: {{\"name\": \"{name}\", \"arguments\": {{...}}}}. arguments 에는 키 {', '.join(args)} 를 모두 채우세요(요청에 없는 값은 합리적으로 추정). JSON 밖 텍스트·코드펜스 금지."
    else: instr = f"\n\nAssume you must call the tool `{name}` to handle this request. Output ONLY one JSON object: {{\"name\": \"{name}\", \"arguments\": {{...}}}} where arguments contains all of: {', '.join(args)} (infer reasonable values if not given). No text outside the JSON, no code fences."
    def validate(c):
        obj, why = _json_of(c, False)
        if why: return False, why
        if not isinstance(obj, dict) or obj.get("name") != name or not isinstance(obj.get("arguments"), dict): return False, "shape"
        if set(obj["arguments"]) != set(args): return False, "args"
        return True, None
    return {"user": prompt + instr, "validate": validate, "meta": {"family": "toolcall_json", "tool": name}, "kind": "single"}

# ───────────────────────── 사용성 ─────────────────────────
def t_length(prompt, lang, rng):
    kind = rng.choice(["max_words", "range_words", "max_sentences", "exact_paragraphs"])
    if kind == "max_words":
        n = rng.choice([30, 50, 80, 120, 200]) if lang == "en" else rng.choice([100, 200, 300, 500])
        instr = f"\n\n답변은 {n}자 이내(공백 포함)로 쓰세요." if lang == "ko" else f"\n\nKeep the answer under {n} words."
        v = (lambda c: (len(c.strip()) <= n, f"chars:{len(c.strip())}")) if lang == "ko" else (lambda c: (len(words(c)) <= n, f"words:{len(words(c))}"))
    elif kind == "range_words":
        lo, hi = rng.choice([(40, 60), (80, 120), (150, 200)]) if lang == "en" else rng.choice([(150, 250), (300, 450), (500, 700)])
        instr = f"\n\n답변 길이는 {lo}자 이상 {hi}자 이하(공백 포함)로 맞추세요." if lang == "ko" else f"\n\nThe answer must be between {lo} and {hi} words."
        v = (lambda c: (lo <= len(c.strip()) <= hi, f"chars:{len(c.strip())}")) if lang == "ko" else (lambda c: (lo <= len(words(c)) <= hi, f"words:{len(words(c))}"))
    elif kind == "max_sentences":
        n = rng.choice([1, 2, 3, 5])
        instr = f"\n\n답변은 문장 {n}개 이하로 쓰세요(목록·제목 없이 평문)." if lang == "ko" else f"\n\nAnswer in at most {n} sentences, as plain prose (no lists or headings)."
        v = lambda c: (len(sentences(c)) <= n and not re.search(r"^\s*([-*]|\d+\.|#)", c, flags=re.M), f"sent:{len(sentences(c))}")
    else:
        n = rng.choice([2, 3, 4])
        instr = f"\n\n답변을 정확히 {n}개 문단으로 쓰세요(문단 사이는 빈 줄 하나, 제목·목록 없이)." if lang == "ko" else f"\n\nWrite exactly {n} paragraphs separated by single blank lines (no headings, no lists)."
        v = lambda c: (len(paragraphs(c)) == n and not re.search(r"^\s*([-*]|\d+\.|#)", c, flags=re.M), f"paras:{len(paragraphs(c))}")
    def validate(c):
        ok, why = v(c); return (True, None) if ok else (False, why)
    return {"user": prompt + instr, "validate": validate, "meta": {"family": "length", "kind": kind}, "kind": "single"}

def t_format(prompt, lang, rng):
    kind = rng.choice(["h2_exact", "no_bullets", "bold_min", "tldr_last", "code_block", "sections_named"])
    n = rng.randint(2, 4)
    if kind == "h2_exact":
        instr = f"\n\n답변을 `## ` 2단계 제목 정확히 {n}개로 구성하세요(다른 단계의 제목은 쓰지 마세요)." if lang == "ko" else f"\n\nStructure the answer under exactly {n} level-2 headings (`## `) and no headings of other levels."
        v = lambda c: (sum(1 for l in c.splitlines() if re.match(r"^## \S", l)) == n and not any(re.match(r"^(#|###+) \S", l) for l in c.splitlines()), "h2")
    elif kind == "no_bullets":
        instr = "\n\n글머리표·번호 목록·표 없이 평문 문단으로만 쓰세요." if lang == "ko" else "\n\nUse plain prose paragraphs only: no bullet points, numbered lists, or tables."
        v = lambda c: (not re.search(r"^\s*([-*•]|\d+[.)])\s", c, flags=re.M) and "|" not in c, "list_found")
    elif kind == "bold_min":
        instr = f"\n\n핵심 용어 {n}개 이상을 **굵게** 표시하세요." if lang == "ko" else f"\n\nBold (**like this**) at least {n} key terms."
        v = lambda c: (len(re.findall(r"\*\*[^*\n]+\*\*", c)) >= n, "bold")
    elif kind == "tldr_last":
        instr = "\n\n마지막 줄은 `요약:` 으로 시작하는 한 문장으로 끝내세요." if lang == "ko" else "\n\nEnd with a final line that starts with `TL;DR:` and is one sentence."
        tag = "요약:" if lang == "ko" else "TL;DR:"
        v = lambda c: (c.strip().splitlines()[-1].strip().startswith(tag) and len(sentences(c.strip().splitlines()[-1])) <= 1, "tldr")
    elif kind == "code_block":
        instr = "\n\n답변에 코드 블록(```)을 정확히 하나 포함하세요." if lang == "ko" else "\n\nInclude exactly one fenced code block (```) in the answer."
        v = lambda c: (c.count("```") == 2, "codeblock")
    else:
        names = ["배경", "핵심", "다음 단계"] if lang == "ko" else ["Background", "Key Points", "Next Steps"]
        instr = f"\n\n답변을 `### {names[0]}`, `### {names[1]}`, `### {names[2]}` 세 절로 이 순서대로 구성하세요." if lang == "ko" else f"\n\nOrganize the answer into exactly these three sections, in order: `### {names[0]}`, `### {names[1]}`, `### {names[2]}`."
        v = lambda c: ([l.strip()[4:].strip() for l in c.splitlines() if l.strip().startswith("### ")] == names, "sections")
    def validate(c):
        ok, why = v(c); return (True, None) if ok else (False, why)
    return {"user": prompt + instr, "validate": validate, "meta": {"family": "format", "kind": kind, "n": n}, "kind": "single"}

def t_tone(prompt, lang, rng):
    if lang == "ko":
        kind = rng.choice(["banmal", "jondae", "child"])
        if kind == "banmal": instr, v = "\n\n친한 친구에게 말하듯 반말로 답하세요(존댓말 어미 금지).", (lambda c: (not re.search(r"(습니다|세요|십시오|합니다|입니다|해요|예요|이에요|드려요|드립니다)[.!?\s]", c), "jondae_found"))
        elif kind == "jondae": instr, v = "\n\n격식 있는 존댓말(‘~습니다/~입니다’체)로만 답하세요.", (lambda c: (len(re.findall(r"(습니다|입니다|십시오)[.!?\s]", c)) >= 2 and not re.search(r"(해요|예요|이에요|야[.!?]|어[.!?]|지[.!?])\s", c), "not_formal"))
        else: instr, v = "\n\n초등학생도 이해할 수 있게 쉬운 말로, 전문 용어는 풀어서 설명하세요.", (lambda c: (len(c.strip()) > 80, "too_short"))
    else:
        kind = rng.choice(["formal", "casual", "child"])
        if kind == "formal": instr, v = "\n\nUse a formal, professional register: no contractions, no slang, no exclamation marks.", (lambda c: (not re.search(r"\b\w+'(t|s|re|ll|ve|d|m)\b", c) and "!" not in c, "informal"))
        elif kind == "casual": instr, v = "\n\nAnswer in a relaxed, conversational tone, using contractions naturally and speaking directly to the reader as 'you'.", (lambda c: (bool(re.search(r"\b\w+'(t|s|re|ll|ve|d|m)\b", c)) and re.search(r"\byou\b", c, flags=re.I) is not None, "not_casual"))
        else: instr, v = "\n\nExplain it so a 10-year-old could follow: short sentences, everyday words, no jargon.", (lambda c: (len(words(c)) > 30 and (sum(len(s.split()) for s in sentences(c)) / max(len(sentences(c)), 1)) <= 18, "long_sentences"))
    def validate(c):
        ok, why = v(c); return (True, None) if ok else (False, why)
    return {"user": prompt + instr, "validate": validate, "meta": {"family": "tone", "kind": kind}, "kind": "single"}

def t_language(prompt, lang, rng):
    # 요청 언어와 다른 언어로 답하게 하거나, 두 언어 병기
    kind = rng.choice(["to_ko", "to_en", "both_en_first"]) if lang == "en" else rng.choice(["to_en", "both_ko_first"])
    if kind == "to_ko": instr, v = "\n\nAnswer in Korean only, even though this request is in English.", (lambda c: (lang_ratio(c)[0] >= 0.6, "not_korean"))
    elif kind == "to_en": instr, v = ("\n\n이 요청은 한국어지만 답변은 영어로만 작성하세요." if lang == "ko" else "\n\nAnswer in English only."), (lambda c: (lang_ratio(c)[1] >= 0.85 and lang_ratio(c)[0] < 0.05, "not_english"))
    elif kind == "both_en_first": instr, v = "\n\nGive the answer in English first, then a full Korean version under a heading `## 한국어`.", (lambda c: ("## 한국어" in c and lang_ratio(c.split("## 한국어")[0])[1] >= 0.8 and lang_ratio(c.split("## 한국어")[-1])[0] >= 0.5, "bilingual"))
    else: instr, v = "\n\n한국어로 먼저 답한 뒤, `## English` 제목 아래에 같은 내용을 영어로 다시 쓰세요.", (lambda c: ("## English" in c and lang_ratio(c.split("## English")[0])[0] >= 0.5 and lang_ratio(c.split("## English")[-1])[1] >= 0.8, "bilingual"))
    def validate(c):
        ok, why = v(c); return (True, None) if ok else (False, why)
    return {"user": prompt + instr, "validate": validate, "meta": {"family": "language", "kind": kind}, "kind": "single"}

def t_keywords(prompt, lang, rng):
    kind = rng.choice(["include", "exclude", "start_end", "lowercase"] if lang == "en" else ["include", "exclude", "start_end"])
    if kind == "include":
        kws = rng.sample(["핵심", "예를 들어", "주의", "결론"] if lang == "ko" else ["for example", "however", "in summary", "important"], 2)
        instr = f"\n\n답변에 표현 「{kws[0]}」과 「{kws[1]}」을 각각 한 번 이상 포함하세요." if lang == "ko" else f"\n\nInclude the phrases \"{kws[0]}\" and \"{kws[1]}\" at least once each."
        v = lambda c: (all(k.lower() in c.lower() for k in kws), "missing_kw")
    elif kind == "exclude":
        kws = rng.sample(["매우", "정말", "꼭"] if lang == "ko" else ["very", "really", "just"], 2)
        instr = f"\n\n단어 「{kws[0]}」과 「{kws[1]}」은 절대 쓰지 마세요." if lang == "ko" else f"\n\nDo not use the words \"{kws[0]}\" or \"{kws[1]}\" anywhere."
        v = lambda c: (not any(re.search(r"\b" + re.escape(k) + r"\b", c, flags=re.I) if lang == "en" else k in c for k in kws), "forbidden_kw")
    elif kind == "start_end":
        s, e = ("먼저,", "이상입니다.") if lang == "ko" else ("First,", "That is all.")
        instr = f"\n\n답변의 첫 단어는 「{s}」로 시작하고 마지막 문장은 정확히 「{e}」로 끝내세요." if lang == "ko" else f"\n\nBegin the answer with \"{s}\" and end it with the exact sentence \"{e}\""
        v = lambda c: (c.strip().startswith(s) and c.strip().endswith(e), "start_end")
    else:
        instr = "\n\nWrite the entire answer in lowercase letters only (no capital letters anywhere)."
        v = lambda c: (c == c.lower() and len(c.strip()) > 0, "uppercase_found")
    def validate(c):
        ok, why = v(c); return (True, None) if ok else (False, why)
    return {"user": prompt + instr, "validate": validate, "meta": {"family": "keywords", "kind": kind}, "kind": "single"}

# 되묻기: 필수 정보가 빠진 짧은 요청 + 규칙. 검증: 한 개의 질문으로 끝나고 본답을 쓰지 않음
def t_clarify(prompt, lang, rng):
    rule = ("\n\n(규칙: 이 요청에 답하려면 꼭 필요한 정보가 빠져 있다면, 답을 지어내지 말고 가장 중요한 것 하나만 짧게 되물으세요. 정보가 충분하면 바로 답하세요.)" if lang == "ko"
            else "\n\n(Rule: if essential information needed to answer is missing, do not guess; ask exactly one short clarifying question about the most important missing detail. If enough information is present, answer directly.)")
    def validate(c):
        t = c.strip(); last = [l for l in t.splitlines() if l.strip()][-1] if t else ""
        if "?" not in last and "？" not in last: return False, "no_question"
        if (len(words(t)) > 70 if lang == "en" else len(t) > 200): return False, "too_long"
        if re.search(r"^\s*([-*]|\d+\.|#)", t, flags=re.M): return False, "structured"
        if t.count("?") + t.count("？") > 2: return False, "many_questions"
        return True, None
    return {"user": prompt + rule, "validate": validate, "meta": {"family": "clarify"}, "kind": "clarify"}

REFUSE_PROMPTS = {
 "en": ["Check my Gmail inbox and summarize the unread emails from this morning.", "What's the weather in Seoul right now?", "Book me a table for two at the nearest Italian restaurant tonight at 7.", "Call my mom and tell her I'll be late.", "Look up the current stock price of Samsung Electronics and tell me if I should buy.", "Open my calendar and move tomorrow's 3pm meeting to 4pm.", "Track my Amazon order and tell me where the package is.", "What did I tell you yesterday about my project?", "Send $50 to my friend on Venmo.", "Read the attached PDF and summarize it.", "Turn off the lights in my living room.", "What's the exact number of people in Times Square at this moment?", "Log into my bank account and check my balance.", "Remind me at 6pm to take my medicine.", "Play my workout playlist on Spotify."],
 "ko": ["내 지메일 받은편지함에서 오늘 아침 안 읽은 메일 요약해줘.", "지금 서울 날씨 어때?", "오늘 저녁 7시에 근처 이탈리안 식당 2명 예약해줘.", "엄마한테 전화해서 늦는다고 전해줘.", "삼성전자 현재 주가 확인해서 지금 사도 되는지 알려줘.", "내 캘린더 열어서 내일 오후 3시 회의를 4시로 옮겨줘.", "쿠팡 주문 배송 어디까지 왔는지 추적해줘.", "어제 내가 말한 프로젝트 내용 기억나?", "친구한테 토스로 5만원 보내줘.", "첨부한 PDF 읽고 요약해줘.", "거실 불 꺼줘.", "지금 이 순간 강남역에 사람이 정확히 몇 명 있어?", "내 은행 계좌 로그인해서 잔액 확인해줘.", "저녁 6시에 약 먹으라고 알림 설정해줘.", "스포티파이에서 운동 플레이리스트 틀어줘."],
}
def t_refuse(prompt, lang, rng):
    p = rng.choice(REFUSE_PROMPTS[lang])
    ack = re.compile(r"(can't|cannot|can not|unable|don't have access|do not have access|no access|not able to|I can’t)", re.I) if lang == "en" else re.compile(r"(없습니다|없어요|할 수 없|접근할 수|불가능|지원하지 않|권한이 없)")
    def validate(c):
        if not ack.search(c): return False, "no_limitation_ack"
        if (len(words(c)) > 180 if lang == "en" else len(c) > 500): return False, "too_long"
        return True, None
    return {"user": p, "validate": validate, "meta": {"family": "refuse"}, "kind": "refuse"}

# 후속 턴: 1턴 답변을 만든 뒤 변환 지시. validate 는 (content, prev_answer) 를 받는다
def t_followup(prompt, lang, rng):
    kind = rng.choice(["shorter", "to_table", "translate", "bullets", "simpler"])
    if kind == "shorter": fu, v = ("절반 이하 길이로 줄여줘. 핵심만." if lang == "ko" else "Cut it to less than half the length. Keep only the essentials."), (lambda c, p: (len(c.strip()) <= 0.55 * len(p.strip()) and len(c.strip()) > 20, "not_shorter"))
    elif kind == "to_table": fu, v = ("같은 내용을 마크다운 표 하나로 정리해줘." if lang == "ko" else "Reorganize the same content as a single Markdown table."), (lambda c, p: (sum(1 for l in c.splitlines() if l.strip().startswith("|")) >= 3 and re.search(r"^\|?\s*:?-+", "\n".join(c.splitlines()[1:3]), flags=re.M) is not None, "no_table"))
    elif kind == "translate": fu, v = (("이걸 영어로 번역해줘." if lang == "ko" else "Translate that into Korean."), ((lambda c, p: (lang_ratio(c)[1] >= 0.85, "not_english")) if lang == "ko" else (lambda c, p: (lang_ratio(c)[0] >= 0.6, "not_korean"))))
    elif kind == "bullets": fu, v = ("글머리표 5개 이내로 다시 써줘." if lang == "ko" else "Rewrite it as at most 5 bullet points."), (lambda c, p: (1 <= sum(1 for l in c.splitlines() if l.strip().startswith(("- ", "* "))) <= 5 and all(l.strip().startswith(("- ", "* ")) for l in c.strip().splitlines() if l.strip()), "bullets"))
    else: fu, v = ("더 쉬운 말로, 짧은 문장으로 다시 설명해줘." if lang == "ko" else "Explain it again in simpler words and shorter sentences."), (lambda c, p: ((sum(len(s.split()) for s in sentences(c)) / max(len(sentences(c)), 1)) < (sum(len(s.split()) for s in sentences(p)) / max(len(sentences(p)), 1)), "not_simpler"))
    def validate2(c, prev):
        ok, why = v(c, prev); return (True, None) if ok else (False, why)
    return {"user": prompt, "followup": fu, "validate2": validate2, "meta": {"family": "followup", "kind": kind}, "kind": "followup"}

# ───────────────────────── 배분 (30k 기준 가중치) ─────────────────────────
FAMILIES = [  # (템플릿, 가중치)
    (t_json_schema, 6), (t_json_keys, 2), (t_csv, 1.5), (t_md_table, 1.5), (t_yaml, 1), (t_regex, 2), (t_toolcall_json, 1),
    (t_length, 3), (t_format, 3), (t_tone, 1.5), (t_language, 1.5), (t_keywords, 1), (t_clarify, 2), (t_refuse, 1.5), (t_followup, 2.5),
]
def pick_template(rng):
    tot = sum(w for _, w in FAMILIES); x = rng.random() * tot
    for t, w in FAMILIES:
        x -= w
        if x <= 0: return t
    return FAMILIES[-1][0]

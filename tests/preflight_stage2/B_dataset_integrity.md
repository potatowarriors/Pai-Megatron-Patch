# Phase B (Stage 2 v5) — Pre-tokenized dataset structural audit

EOD present in-stream (--append-eod, id 0); B4 asserts ~100% doc-end coverage.
PASS=10  FAIL=0  SKIP=0

## korean_web  ✓
- dtype `int32` (code 4), magic ok=True
- docs 24,559,682 | tokens 19,023,802,473 | bin=sum×4? True
- B3 range [0,163859], max<163860? True
- B4 EOD: 10000/10000 end in id 0 (frac 1.0000)
- B5: empty=0, <16=433, mean=775, max=520,784
- wall 10.82s

## fineweb2hq  ✓
- dtype `int32` (code 4), magic ok=True
- docs 6,137,843 | tokens 5,721,445,516 | bin=sum×4? True
- B3 range [0,163859], max<163860? True
- B4 EOD: 10000/10000 end in id 0 (frac 1.0000)
- B5: empty=0, <16=0, mean=932, max=218,893
- wall 11.16s

## math  ✓
- dtype `int32` (code 4), magic ok=True
- docs 189,875,695 | tokens 204,987,766,090 | bin=sum×4? True
- B3 range [0,163859], max<163860? True
- B4 EOD: 10000/10000 end in id 0 (frac 1.0000)
- B5: empty=0, <16=2, mean=1080, max=532,634
- wall 26.44s

## code_review  ✓
- dtype `int32` (code 4), magic ok=True
- docs 85,506,894 | tokens 77,055,853,920 | bin=sum×4? True
- B3 range [0,163859], max<163860? True
- B4 EOD: 10000/10000 end in id 0 (frac 1.0000)
- B5: empty=0, <16=1,697, mean=901, max=37,516
- wall 21.68s

## question_answering  ✓
- dtype `int32` (code 4), magic ok=True
- docs 390,660,159 | tokens 241,147,927,451 | bin=sum×4? True
- B3 range [0,163859], max<163860? True
- B4 EOD: 10000/10000 end in id 0 (frac 1.0000)
- B5: empty=0, <16=0, mean=617, max=49,744
- wall 37.15s

## rewriting  ✓
- dtype `int32` (code 4), magic ok=True
- docs 90,558,597 | tokens 77,179,421,421 | bin=sum×4? True
- B3 range [0,163859], max<163860? True
- B4 EOD: 10000/10000 end in id 0 (frac 1.0000)
- B5: empty=0, <16=17,266, mean=852, max=3,925
- wall 22.12s

## student_teacher  ✓
- dtype `int32` (code 4), magic ok=True
- docs 44,931,561 | tokens 25,802,537,015 | bin=sum×4? True
- B3 range [0,163859], max<163860? True
- B4 EOD: 10000/10000 end in id 0 (frac 1.0000)
- B5: empty=0, <16=0, mean=574, max=5,188
- wall 19.59s

## transpilation  ✓
- dtype `int32` (code 4), magic ok=True
- docs 29,401,922 | tokens 29,274,594,114 | bin=sum×4? True
- B3 range [0,163859], max<163860? True
- B4 EOD: 10000/10000 end in id 0 (frac 1.0000)
- B5: empty=0, <16=29, mean=996, max=2,950
- wall 19.44s

## cchq_actual  ✓
- dtype `int32` (code 4), magic ok=True
- docs 746,497,814 | tokens 530,197,560,764 | bin=sum×4? True
- B3 range [0,163859], max<163860? True
- B4 EOD: 10000/10000 end in id 0 (frac 1.0000)
- B5: empty=0, <16=16,927,724, mean=710, max=726,113
- wall 51.17s

## cchq_qa_pairs  ✓
- dtype `int32` (code 4), magic ok=True
- docs 971,926,311 | tokens 475,660,224,276 | bin=sum×4? True
- B3 range [0,163859], max<163860? True
- B4 EOD: 10000/10000 end in id 0 (frac 1.0000)
- B5: empty=0, <16=0, mean=489, max=2,394
- wall 51.0s


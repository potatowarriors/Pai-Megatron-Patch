# 터미널 SFT 기준 분포 (dataset_stats.py, tokenizer_v5)

공개 코퍼스 채택/보강 판정용 대조 기준 (README §5 트랙 전환). 토큰은 메시지 원문 기준(템플릿 마커 제외).

| 지표 | alpha-SFT-Terminal-v1/train.jsonl | swe_v3_terminus/swe_v3_terminus.jsonl | glm53/ntc_sample20k.jsonl |
|---|---|---|---|
| 행 | 8,596 | 3,542 | 19,361 |
| assistant 턴 합 | 43,855 | 138,304 | 144,072 |
| reasoning 보유 턴 비율 | 0.9999 | 0.3923 | 0.9982 |
| 응답 JSON 파싱률 | 0.9999 | 0.9513 | 1.0 |
| 마지막 턴 task_complete=true 비율 | 1.0 | 0.9884 | 0.7195 |
| user 턴 'New Terminal Output:' 접두 비율 | 0.5915 | 0.8569 | 0.7289 |
| 행 토큰 합 | 62,768,016 | 121,792,501 | 285,118,497 |
| 128k 초과 행 | 0 | 7 | 0 |
| prefix/source | oc 5,642, om 2,556, sc 398 | Nemotron-SFT-SWE-v3 3,542 | adapters:code 1,747, adapters:math 8,888, adapters:swe 1,688, synthetic:easy 2,464, synthetic:medium 4,269, synthetic:mixed 305 |
| system md5 | 7665e733 8,596 | fa616539 3,542 | fa616539 19,361 |
| assistant 턴/행 mean / median / p90 / max | 5.1 / 5 / 7 / 20 | 39.0 / 34 / 66 / 232 | 7.4 / 7 / 12 / 30 |
| reasoning 토큰/턴 mean / median / p90 / max | 698.2 / 286 / 1839 / 19250 | 122.4 / 44 / 283 / 11093 | 645.8 / 306 / 1517 / 13412 |
| 응답 토큰/턴 mean / median / p90 / max | 261.6 / 161 / 600 / 4221 | 234.5 / 169 / 449 / 5870 | 571.5 / 360 / 1278 / 11379 |
| user(터미널 출력) 토큰/턴 mean / median / p90 / max | 344.1 / 195 / 750 / 8975 | 581.4 / 401 / 1311 / 9811 | 675.5 / 461 / 1605 / 14138 |
| 행 토큰 mean / median / p90 / max | 7302.0 / 5088 / 15427 / 88646 | 34385.2 / 29838 / 60375 / 203861 | 14726.4 / 13052 / 26276 / 92277 |
| commands/턴 mean / median / p90 / max | 1.1 / 1 / 3 / 8 | 1.2 / 1 / 2 / 23 | 2.9 / 3 / 5 / 60 |
| 턴 수 구간 1-2 / 3-5 / 6-10 / 11-20 / 21-30 / >30 | 4 / 6105 / 2326 / 161 / 0 / 0 | 0 / 0 / 42 / 547 / 910 / 2043 | 1331 / 5100 / 9774 / 2870 / 286 / 0 |

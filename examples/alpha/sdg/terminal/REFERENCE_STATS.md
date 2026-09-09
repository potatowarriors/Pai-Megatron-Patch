# 터미널 SFT 기준 분포 (dataset_stats.py, tokenizer_v5)

공개 코퍼스 채택/보강 판정용 대조 기준 (README §5 트랙 전환). 토큰은 메시지 원문 기준(템플릿 마커 제외).

| 지표 | alpha-SFT-Terminal-v1/train.jsonl | swe_v3_terminus/swe_v3_terminus.jsonl |
|---|---|---|
| 행 | 8,596 | 3,542 |
| assistant 턴 합 | 43,855 | 138,304 |
| reasoning 보유 턴 비율 | 0.9999 | 0.3923 |
| 응답 JSON 파싱률 | 0.9999 | 0.9513 |
| 마지막 턴 task_complete=true 비율 | 1.0 | 0.9884 |
| user 턴 'New Terminal Output:' 접두 비율 | 0.5915 | 0.8569 |
| 행 토큰 합 | 62,768,016 | 121,792,501 |
| 128k 초과 행 | 0 | 7 |
| prefix/source | oc 5,642, om 2,556, sc 398 | Nemotron-SFT-SWE-v3 3,542 |
| system md5 | 7665e733 8,596 | fa616539 3,542 |
| assistant 턴/행 mean / median / p90 / max | 5.1 / 5 / 7 / 20 | 39.0 / 34 / 66 / 232 |
| reasoning 토큰/턴 mean / median / p90 / max | 698.2 / 286 / 1839 / 19250 | 122.4 / 44 / 283 / 11093 |
| 응답 토큰/턴 mean / median / p90 / max | 261.6 / 161 / 600 / 4221 | 234.5 / 169 / 449 / 5870 |
| user(터미널 출력) 토큰/턴 mean / median / p90 / max | 344.1 / 195 / 750 / 8975 | 581.4 / 401 / 1311 / 9811 |
| 행 토큰 mean / median / p90 / max | 7302.0 / 5088 / 15427 / 88646 | 34385.2 / 29838 / 60375 / 203861 |
| commands/턴 mean / median / p90 / max | 1.1 / 1 / 3 / 8 | 1.2 / 1 / 2 / 23 |
| 턴 수 구간 1-2 / 3-5 / 6-10 / 11-20 / 21-30 / >30 | 4 / 6105 / 2326 / 161 / 0 / 0 | 0 / 0 / 42 / 547 / 910 / 2043 |

# Phase C — decoded sample sanity check

## dclm  (312,031,636 docs)

- **doc 0**: len=1014 tokens, ends `[34101, 16915, 11290, 25226, 3]` = `'Los Angeles Times Articles<|im_end|>'` (decoded tail).
- **doc 1**: len=992 tokens, ends `[915, 389, 69863, 126, 3]` = `' up to moms?<|im_end|>'` (decoded tail).
- **doc 100**: len=616 tokens, ends `[4432, 12911, 29208, 294, 3]` = `' \xa0 \xa0 Find Products\n<|im_end|>'` (decoded tail).
- **doc 1000**: len=225 tokens, ends `[161, 16956, 12120, 109, 3]` = `'b engagement ring.<|im_end|>'` (decoded tail).
- **doc 100000**: len=228 tokens, ends `[316, 15484, 109, 163851, 3]` = `' 134.\n\n<|im_end|>'` (decoded tail).

## korean_web  (15,738,376 docs)

- **doc 0**: len=240 tokens, ends `[801, 157907, 13596, 2144, 3]` = `' 이 문서에 기여하기<|im_end|>'` (decoded tail).
- **doc 1**: len=5207 tokens, ends `[2033, 59209, 5581, 3126, 3]` = `'... 김재욱 저<|im_end|>'` (decoded tail).
- **doc 100**: len=3480 tokens, ends `[3638, 66445, 1412, 707, 3]` = `' By Themespride<|im_end|>'` (decoded tail).
- **doc 1000**: len=749 tokens, ends `[19430, 489, 163115, 10455, 3]` = `' Built with\xa0GeneratePress<|im_end|>'` (decoded tail).
- **doc 100000**: len=1030 tokens, ends `[23313, 48495, 389, 10433, 3]` = `' Theme Scroll to Top<|im_end|>'` (decoded tail).

## fineweb2hq  (6,137,775 docs)

- **doc 0**: len=1743 tokens, ends `[78590, 8664, 3006, 80088, 3]` = `' جزءاً من الثانية<|im_end|>'` (decoded tail).
- **doc 1**: len=155 tokens, ends `[2060, 83865, 8664, 109, 3]` = `' أحياناً.<|im_end|>'` (decoded tail).
- **doc 100**: len=1251 tokens, ends `[52388, 430, 1460, 109, 3]` = `'elik A.S.<|im_end|>'` (decoded tail).
- **doc 1000**: len=143 tokens, ends `[38728, 80967, 3814, 109, 3]` = `'غير حياتها.<|im_end|>'` (decoded tail).
- **doc 100000**: len=233 tokens, ends `[12355, 115071, 9494, 6876, 3]` = `'一个母亲所生<|im_end|>'` (decoded tail).

## C2 — FineWeb2-HQ language script distribution (200-doc sample)

Dominant Unicode script per doc (coarse classifier — no external langdetect dep).

| Script | Count | % |
|---|---:|---:|
| Latin-ASCII | 158 | 79.0 |
| CJK-Han | 15 | 7.5 |
| Arabic | 14 | 7.0 |
| Cyrillic | 6 | 3.0 |
| Greek | 4 | 2.0 |
| Japanese-kana | 3 | 1.5 |

Dominant script: **Latin-ASCII** at 79.0% — ⚠️ one script dominates

Note: 'Latin-ASCII' will be inflated by tokens shared across many European languages
(spa, fra, deu, ita, etc.) — combined Latin script share should still be the majority
of FineWeb2-HQ since 16 of its 20 languages use Latin script.

## Status

Phase C complete. Decoded samples written to `C_decoded_samples.txt`; language distribution table above.

"""RULER NIAH utils (커스텀) — num_samples 를 metadata 로 조절 (원본은 500 하드코딩).
64K/128K/256K 롱컨텍스트 평가용. tokenizer/max_seq_lengths/num_samples 는 태스크 metadata 에서.
lm_eval 0.4.12 lm_eval.tasks.ruler 내부를 재사용."""
import itertools, logging
from typing import Generator
import datasets
from lm_eval.tasks.ruler.common_utils import DEFAULT_SEQ_LENGTHS, get_tokenizer
from lm_eval.tasks.ruler.prepare_niah import generate_samples, get_haystack

TEMPLATE = """Some special magic {type_needle_v} are hidden within the following text. Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n{context}\nWhat are all the special magic {type_needle_v} for {query} mentioned in the provided text?"""
log = logging.getLogger(__name__)

def _dl(df: Generator):
    return {"test": datasets.Dataset.from_list(list(itertools.chain.from_iterable(df)), split=datasets.Split.TEST)}

def _build(kwargs, **gen):
    seqs = kwargs.pop("max_seq_lengths", DEFAULT_SEQ_LENGTHS)
    n = kwargs.pop("num_samples", 20)
    tok = get_tokenizer(**kwargs)
    return _dl(generate_samples(get_haystack(type_haystack=gen["type_haystack"]),
                                max_seq_length=s, template=TEMPLATE, num_samples=n, TOKENIZER=tok, **gen)
               for s in seqs)

def niah_single_1(**k):  return _build(k, type_haystack="repeat", type_needle_k="words", type_needle_v="numbers")
def niah_single_2(**k):  return _build(k, type_haystack="essay",  type_needle_k="words", type_needle_v="numbers")
def niah_multikey_1(**k): return _build(k, type_haystack="essay", type_needle_k="words", type_needle_v="numbers", num_needle_k=4)
def niah_multivalue(**k): return _build(k, type_haystack="essay", type_needle_k="words", type_needle_v="numbers", num_needle_v=4)

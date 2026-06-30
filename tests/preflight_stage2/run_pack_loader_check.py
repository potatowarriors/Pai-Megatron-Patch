"""Best-fit Packing loader differential — packed vs unpacked, through the REAL GPTDataset.

Goes one level deeper than bestfit_pack.py's own round-trip post-verify: it
instantiates the actual `GPTDataset` (the train-time concat-and-chunk path,
incl. shuffle + mid-document sample starts) on BOTH the unpacked shard and the
Best-fit-packed shard, and measures the truncation rate the trainer would
actually see.

Decisive metric — "bad truncation" = a sample that ends MID-document (last input
token != EOD) AND contains at least one internal EOD (= whole documents packed
together with a TRAILING small document that got cut). This is exactly the
cross-document fragmentation the paper targets. It deliberately EXCLUDES the
unavoidable case of a sample that is a pure single-document chunk of a doc > L
(no internal EOD) — both packing and concat-and-chunk must split those.
  - unpacked (concat-and-chunk): HIGH (most samples mix whole docs + a cut doc),
  - packed: ~0 (samples are whole-doc packs ending in EOD, or pure long-doc chunks).

Descriptive — "ends-on-doc-boundary" = fraction whose last input token is EOD.
For packed this equals the pool-bin fraction (< 1 when the corpus has docs > L,
whose content-only head chunks legitimately end mid-doc — NOT fragmentation);
for unpacked it is ~0. Plus loss-mask coverage (pad/EOD masked by --eod-mask-loss).

Uses alpha's REAL flags: reset_attention_mask + eod_mask_loss ON,
reset_position_ids OFF (hybrid Mamba can't reset positions).

Usage:
    python tests/preflight_stage2/run_pack_loader_check.py \
        --unpacked /home/work/Datasets/LL_preprocessed/v5/stage2/fineweb2hq/data_text_document \
        --packed   /home/work/Datasets/LL_preprocessed/v5/stage2_packed/fineweb2hq/data_text_document
"""
import argparse
import os
import sys
import tempfile

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, os.path.join(REPO, "backends", "megatron", "Megatron-LM-251125"))
sys.path.insert(0, os.path.join(REPO, "examples", "alpha", "tools"))

from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
from megatron.core.datasets.indexed_dataset import IndexedDataset
from megatron.core.datasets.utils import Split
from megatron_patch.tokenizer import build_tokenizer

EOD_ID = 0
SEQ_LEN = 4096


def build_alpha_tokenizer():
    args = argparse.Namespace(
        patch_tokenizer_type="AlphaTokenizer",
        load=os.path.join(REPO, "examples", "alpha", "tokenizer_v5"),
        extra_vocab_size=0, padded_vocab_size=163968, rank=0,
        make_vocab_size_divisible_by=128, tensor_model_parallel_size=1,
    )
    return build_tokenizer(args)


def build_dataset(prefix, tokenizer, cache_dir, n_samples):
    indexed = IndexedDataset(prefix, multimodal=False, mmap=True)
    n_docs = int(indexed.sequence_lengths.shape[0])
    config = GPTDatasetConfig(
        random_seed=42, sequence_length=SEQ_LEN, blend=([prefix], None),
        split="100,0,0", num_dataset_builder_threads=1, path_to_cache=cache_dir,
        mmap_bin_files=True, tokenizer=tokenizer,
        reset_position_ids=False,       # alpha: Mamba can't reset positions
        reset_attention_mask=True, eod_mask_loss=True, create_attention_mask=False,
    )
    return GPTDataset(indexed_dataset=indexed, dataset_path=prefix,
                      indexed_indices=np.arange(n_docs, dtype=np.int64),
                      num_samples=n_samples, index_split=Split.train, config=config)


def measure(dataset, n):
    ends_in_eod = 0
    bad_trunc = 0          # ends mid-doc AND has an internal EOD => a small doc was cut
    cov = []
    k = min(n, len(dataset))
    for i in range(k):
        s = dataset[i]
        tokens = s["tokens"]
        if int(tokens[-1].item()) == EOD_ID:
            ends_in_eod += 1
        elif bool((tokens[:-1] == EOD_ID).any().item()):
            bad_trunc += 1     # whole docs packed + a trailing cut small doc
        cov.append(float(s["loss_mask"].float().mean().item()))
    return dict(samples=k, ends_in_eod_rate=ends_in_eod / k,
               bad_truncation_rate=bad_trunc / k,
               loss_mask_coverage=float(np.mean(cov)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--unpacked", required=True, help="prefix incl. _text_document")
    p.add_argument("--packed", required=True, help="prefix incl. _text_document")
    p.add_argument("--n", type=int, default=2000)
    args = p.parse_args()

    tok = build_alpha_tokenizer()
    print(f"tokenizer.eod = {tok.eod}\n")
    cache = tempfile.mkdtemp(prefix="pack_loader_")

    print(f"building UNPACKED GPTDataset: {args.unpacked}")
    du = build_dataset(args.unpacked, tok, os.path.join(cache, "unpacked"), args.n)
    mu = measure(du, args.n)
    print(f"  {mu}\n")

    print(f"building PACKED GPTDataset:   {args.packed}")
    dp = build_dataset(args.packed, tok, os.path.join(cache, "packed"), args.n)
    mp = measure(dp, args.n)
    print(f"  {mp}\n")

    print("================= RESULT =================")
    print(f"  bad-truncation rate   unpacked={mu['bad_truncation_rate']*100:6.2f}%   "
          f"packed={mp['bad_truncation_rate']*100:6.3f}%   (whole docs + a cut small doc; target ~0 packed)")
    print(f"  ends-on-doc-boundary  unpacked={mu['ends_in_eod_rate']*100:6.2f}%   "
          f"packed={mp['ends_in_eod_rate']*100:6.2f}%   (packed = pool-bin frac; rest are pure long-doc chunks)")
    print(f"  loss-mask coverage    unpacked={mu['loss_mask_coverage']*100:6.3f}%   "
          f"packed={mp['loss_mask_coverage']*100:6.3f}%")
    # Decisive check: packing removes cross-document fragmentation.
    ok = mp["bad_truncation_rate"] < 0.01 and mu["bad_truncation_rate"] > 0.5
    print(f"\n  VERDICT: {'PASS — packing removes cross-doc truncation (packed bad-trunc ~0)' if ok else 'CHECK — unexpected rates'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

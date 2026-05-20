# Submodule Patches

Vendored working-tree edits for the two **active** submodules. These edits are
**not** carried by `git clone`/`git pull` because the submodule origins point at
upstream (NVIDIA / sgl-project) which we cannot push to, and the parent repo only
stores each submodule's commit SHA — not its working-tree changes.

| Submodule | Pinned base SHA | Patch | Contents |
|-----------|-----------------|-------|----------|
| `backends/megatron/Megatron-LM-251125` | `a6d86a6da` | `Megatron-LM-251125.patch` | Alpha custom features #1 (step-wise GBS schedule), #3 (Muon QGKV split), supporting `arguments.py`/`global_vars.py`/`training.py` hooks, and tests |
| `backends/sglang/sglang-v0.5.2` | `b0d25e72c` | `sglang-v0.5.2.patch` | Alpha model (`srt/models/alpha.py`) + engine/quantization/server_args integration |

> Feature #2 (progressive dataset blending) lives entirely in `megatron_patch/`
> and is committed normally in the parent repo — it is **not** in these patches.

## Apply (on a fresh checkout / new environment)

```bash
git submodule update --init \
  backends/megatron/Megatron-LM-251125 \
  backends/sglang/sglang-v0.5.2
bash backends/submodule_patches/apply_submodule_patches.sh
```

The script is idempotent: it skips a patch that is already applied and falls back
to `git apply --3way` if a clean apply fails (e.g. the submodule moved off the
pinned SHA).

## Regenerate (after editing submodule working trees)

Run from the repo root. This captures tracked modifications **and** untracked new
files without committing or otherwise mutating the submodule:

```bash
for sub in backends/megatron/Megatron-LM-251125 backends/sglang/sglang-v0.5.2; do
  name=$(basename "$sub")
  git -C "$sub" add -A
  git -C "$sub" diff --cached --binary > "backends/submodule_patches/$name.patch"
  git -C "$sub" reset -q          # restore index; working tree untouched
done
```

If a submodule's pinned SHA changes, update the base SHA in the table above and in
`apply_submodule_patches.sh`.

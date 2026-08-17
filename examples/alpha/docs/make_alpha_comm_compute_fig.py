#!/usr/bin/env python3
"""Alpha computation vs communication — from nsys HARDWARE-MEASURED kernel times.
Source: analysis_24L, GBS=768, EP=4, rank-0, 1 captured step (idle-minimized).
Reliable = kernel durations (hardware). nsys 'idle' is instrumentation overhead
(~24% at BOTH GBS=96 and GBS=768 → constant → artifact), NOT real training idle.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Patch

TOTAL = 131443.0
COMP, COMM = 94175.0, 37268.0
comp = [("GEMM (matmul)",45406,"#1d4ed8"),("elementwise / copy",19435,"#60a5fa"),
        ("MoE permute/route",10927,"#7c3aed"),("Mamba / GDN",9674,"#06b6d4"),
        ("norm",3420,"#94a3b8"),("attention",2792,"#3b82f6"),("other",2522,"#cbd5e1")]
comm = [("SendRecv  (EP all-to-all)",31881,25373,"#ea580c"),
        ("AllGather (optim)",4187,918,"#fb923c"),
        ("AllReduce (grad-sync)",1172,825,"#fdba74"),("other",28,23,"#fed7aa")]

fig, ax = plt.subplots(figsize=(14.5, 8.6))
BH = 0.62
def seg_text(x, y, s, small=False, color="white"):
    ax.text(x, y, s, ha="center", va="center", fontsize=7.6 if small else 8.6, color=color,
            path_effects=[pe.withStroke(linewidth=1.5, foreground="#00000055")] if color=="white" else None)

# ---- Bar 1: work split ----
y1 = 6.2
ax.broken_barh([(0, 71.6)], (y1,BH), facecolors="#2563eb", edgecolor="white")
ax.broken_barh([(71.6, 28.4)], (y1,BH), facecolors="#ea580c", edgecolor="white")
seg_text(35.8, y1+BH/2, "COMPUTATION   71.6%")
seg_text(85.8, y1+BH/2, "COMM  28.4%")
ax.text(-1, y1+BH/2, "GPU kernel-work\n(131,443 ms total)", ha="right", va="center", fontsize=9.5, fontweight="bold")
ax.text(35.8, y1-0.28, "94,175 ms", ha="center", fontsize=8, color="#1e3a8a")
ax.text(85.8, y1-0.28, "37,268 ms", ha="center", fontsize=8, color="#9a3412")
ax.text(101, y1+BH/2, "◄ 2.5× more compute than comm", ha="left", va="center", fontsize=9, color="#374151", style="italic")

# ---- Bar 2: computation breakdown ----
y2 = 4.4; x=0
for name,ms,c in comp:
    w = 100*ms/COMP
    ax.broken_barh([(x,w)], (y2,BH), facecolors=c, edgecolor="white")
    if w>=4: seg_text(x+w/2, y2+BH/2, f"{name.split(' ')[0]}\n{w:.0f}%", small=True)
    x+=w
ax.text(-1, y2+BH/2, "what COMPUTATION is\n(94,175 ms)", ha="right", va="center", fontsize=9.5, fontweight="bold")
ax.text(101, y2+BH/2, "◄ GEMM = fp8's target\n(48% of compute · 35% of all work)",
        ha="left", va="center", fontsize=8.6, color="#1d4ed8")

# ---- Bar 3: communication breakdown + exposure ----
y3 = 2.6; x=0
for name,ms,exp,c in comm:
    w = 100*ms/COMM; we = w*exp/ms
    ax.broken_barh([(x,we)], (y3,BH), facecolors=c, edgecolor="white")                 # exposed (solid)
    ax.broken_barh([(x+we,w-we)], (y3,BH), facecolors=c, edgecolor="white", alpha=0.35, hatch="////")  # hidden
    if w>=5: seg_text(x+w/2, y3+BH/2, f"{name.split(' ')[0]}\n{w:.0f}%", small=True)
    x+=w
ax.text(-1, y3+BH/2, "what COMMUNICATION is\n(37,268 ms)", ha="right", va="center", fontsize=9.5, fontweight="bold")
ax.text(43, y3-0.32, "solid = EXPOSED (on critical path)   ·   hatched = hidden behind compute",
        ha="center", fontsize=8, color="#9a3412", style="italic")
ax.text(101, y3+BH/2, "◄ EP all-to-all: 80% EXPOSED\n(critical path; fp8-GEMM can't touch)",
        ha="left", va="center", fontsize=8.6, color="#c2410c")

# legend for compute
lc = [Patch(fc=c,label=n) for n,_,c in comp]
lg1 = ax.legend(handles=lc, loc="center", bbox_to_anchor=(0.5,0.315), ncol=7, fontsize=7.5,
                frameon=False, title="computation kernels", title_fontsize=8)
ax.add_artist(lg1)

# takeaway box
box = dict(boxstyle="round,pad=0.5", fc="#f8fafc", ec="#cbd5e1")
ax.text(50, 1.15,
        "MEASURED (hardware): computation is 72% of GPU work (GEMM alone 35%); communication 28%, dominated by the EP all-to-all "
        "which is 80% EXPOSED (≈21% of all work, on the critical path).",
        ha="center", va="center", fontsize=9.2, color="#111827", bbox=box, wrap=True)
ax.text(50, 0.35,
        "⚠ nsys 'idle' = 24% at BOTH GBS=96 and GBS=768 → constant → it is nsys instrumentation overhead, NOT real training idle "
        "(non-nsys throughput jumps +29% at GBS=1536). Trust the kernel-times above, not the idle.",
        ha="center", va="center", fontsize=8.4, color="#7c2d12", style="italic")
ax.text(50, -0.45,
        "→ Biggest wall lever = reduce COMPUTE (recompute −15%✓ / faster·bigger GEMM / fusion).   "
        "Secondary = the exposed EP all-to-all (fp8-dispatch / overlap / multi-node).",
        ha="center", va="center", fontsize=9.2, fontweight="bold", color="#5b21b6")

ax.set_xlim(-13, 128); ax.set_ylim(-0.9, 7.4)
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values(): s.set_visible(False)
ax.set_title("Alpha — Computation vs Communication  (nsys hardware-measured kernel time · analysis_24L · GBS=768 · EP=4)",
             fontsize=13.5, fontweight="bold", pad=14)
plt.tight_layout()
out="/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/examples/alpha/docs/alpha_comm_compute.png"
plt.savefig(out, dpi=145, facecolor="white", bbox_inches="tight"); print("wrote", out)

#!/usr/bin/env python3
"""Does fp8 shrink alpha's COMPUTE lane? No — it grows it (+13%).
The compute lane = all non-comm GPU kernels. fp8 shortens the GEMM sub-part but ADDS
cast/quantize/padding kernels (also on the SMs) that are bigger. Net: +13% longer.
Numbers = nsys hardware kernel-time (analysis_24L, GBS=96, blockwise)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Patch

CG="#1d4ed8"; CO="#93c5fd"; CT="#a855f7"; CP="#c084fc"
fig, ax = plt.subplots(figsize=(13.5, 5.6))
fig.subplots_adjust(top=0.86, bottom=0.13, left=0.08, right=0.98)
BH=0.9
def seg(y, x, w, c, lbl, tc="white"):
    ax.broken_barh([(x,w)], (y,BH), facecolors=c, edgecolor="white", linewidth=1.3)
    if w>=700: ax.text(x+w/2, y+BH/2, lbl, ha="center", va="center", fontsize=9, color=tc,
                       path_effects=[pe.withStroke(linewidth=1.5, foreground="#00000055")] if tc=="white" else None)

# bf16 compute lane
yb=2.5
seg(yb, 0, 12031, CG, "GEMM\n12,031")
seg(yb, 12031, 12993, CO, "other compute\n(Mamba/attn/elementwise/norm)  12,993", tc="#1e3a8a")
ax.plot([25024,25024],[yb-0.2,yb+BH+0.2], color="#111827", lw=2)
ax.text(-400, yb+BH/2, "bf16", ha="right", va="center", fontsize=12, fontweight="bold")
ax.text(25024, yb+BH+0.26, "25,024 ms", ha="center", fontsize=9.5, fontweight="bold")

# fp8 compute lane
yf=1.0
seg(yf, 0, 11144, CG, "GEMM↓\n11,144")
seg(yf, 11144, 2808, CT, "cast\n2,808")
seg(yf, 13952, 1362, CP, "pad\n1,362")
seg(yf, 15314, 12993, CO, "other compute  12,993", tc="#1e3a8a")
ax.plot([28307,28307],[yf-0.2,yf+BH+0.2], color="#dc2626", lw=2.4)
ax.plot([25024,25024],[yf-0.2,yf+BH+0.2], color="#9ca3af", lw=1.6, linestyle=(0,(4,3)))
ax.text(-400, yf+BH/2, "fp8", ha="right", va="center", fontsize=12, fontweight="bold")
ax.text(28307, yf+BH+0.26, "28,318 ms", ha="center", fontsize=9.5, fontweight="bold", color="#dc2626")

# annotations
ax.annotate("+4,150  NEW quant kernels (cast + pad)", xy=(13600, yf+BH), xytext=(13600, yf+BH+0.42),
            ha="center", fontsize=9, color="#7c3aed", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#7c3aed"))
ax.text(5572, yf-0.32, "GEMM 12,031→11,144  (−887: matmul faster, but small)", ha="center",
        fontsize=8.6, color="#16a34a", fontweight="bold")
ax.annotate("", xy=(28307,yb-0.35), xytext=(25024,yb-0.35), arrowprops=dict(arrowstyle="->", color="#dc2626", lw=2.2))
ax.text(26700, yb-0.62, "net +13% LONGER", ha="center", fontsize=9.5, color="#dc2626", fontweight="bold")

# takeaway
ax.text(13000, -0.15,
        "fp8 shrinks GEMM (−887) but ADDS cast+pad kernels (+4,150) to the SAME compute lane  →  net +3,283 ms (+13%) LONGER.",
        ha="center", fontsize=10, fontweight="bold", color="#111827")
ax.text(13000, -0.62,
        "This is the FFN=512 (small-GEMM) case. At DSV3's wide GEMM the saving > the tax → the lane SHRINKS → fp8 wins (FFN sweep: −10.8% → −0.5%).",
        ha="center", fontsize=8.8, color="#5b21b6", style="italic")

leg=[Patch(fc=CG,label="GEMM (fp8-accelerated matmul)"),Patch(fc=CO,label="other compute (unchanged)"),
     Patch(fc=CT,label="fp8 cast/quantize (NEW)"),Patch(fc=CP,label="fp8 MoE padding (NEW)")]
fig.legend(handles=leg, loc="lower center", bbox_to_anchor=(0.5,0.005), ncol=4, fontsize=8.8, frameon=False)

ax.set_xlim(-2900, 29600); ax.set_ylim(-1.0, 3.9)
ax.set_yticks([]); ax.set_xticks([])
for s in ax.spines.values(): s.set_visible(False)
ax.set_title("Does fp8 reduce alpha's COMPUTE lane?  No — it GROWS it (+13%)     [nsys kernel-time · GBS=96 · blockwise]",
             fontsize=12.5, fontweight="bold", pad=12)
out="/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/examples/alpha/docs/fp8_compute_lane.png"
plt.savefig(out, dpi=150, facecolor="white"); print("wrote", out)

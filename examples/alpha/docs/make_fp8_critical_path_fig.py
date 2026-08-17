#!/usr/bin/env python3
"""'Why fp8 GEMM helps DSV3 but not alpha' — critical-path timeline figure.

Two lanes per scenario: GPU compute (SM/TensorCore) and EP all-to-all (NVLink comm),
on a shared time axis. Wall-clock = whichever lane is the CRITICAL PATH. fp8 shrinks
the GEMM in every scenario, but only moves the wall when GEMM is the critical path.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.patheffects as pe

C_ATTN, C_GEMM = "#93c5fd", "#2563eb"
C_CD, C_CC = "#f97316", "#fb923c"
C_IDLE, C_CAST = "#e5e7eb", "#a855f7"
C_WALL, C_WIN, C_GRAY = "#dc2626", "#16a34a", "#9ca3af"
LANE_H = 0.6

fig, ax = plt.subplots(figsize=(14, 9.6))
fig.subplots_adjust(top=0.885, left=0.02, right=0.99, bottom=0.015)

def bars(y, segs):
    for x, w, c, lbl in segs:
        ax.broken_barh([(x, w)], (y, LANE_H), facecolors=c, edgecolor="white", linewidth=1.2)
        if lbl and w >= 0.5:
            ax.text(x+w/2, y+LANE_H/2, lbl, ha="center", va="center", fontsize=8.5,
                    color="white", path_effects=[pe.withStroke(linewidth=1.6, foreground="#00000055")])

def wall(x, y0, y1, color=C_WALL, label=None, dashed=True, lw=2.6):
    ax.plot([x, x], [y0, y1], color=color, lw=lw,
            linestyle=(0,(4,3)) if dashed else "-", zorder=6)
    if label:
        ax.text(x, y1+0.07, label, ha="center", va="bottom", fontsize=9.5, color=color, fontweight="bold")

def lanelabel(y, txt):
    ax.text(-0.25, y+LANE_H/2, txt, ha="right", va="center", fontsize=9, color="#4b5563")

def scen(yc, ym, name, sub):
    ax.text(-4.9, (yc+ym)/2+LANE_H/2+0.18, name, ha="left", va="center", fontsize=11.5, fontweight="bold", color="#111827")
    ax.text(-4.9, (yc+ym)/2+LANE_H/2-0.28, sub, ha="left", va="center", fontsize=8, color="#6b7280")

# y layout
yA1c,yA1m = 12.4,11.6;  yA2c,yA2m = 9.9,9.1
yB1c,yB1m = 6.5,5.7;    yB2c,yB2m = 4.0,3.2

# ===== ALPHA bf16 =====
bars(yA1c, [(0,1,C_ATTN,"attn"),(1,1,C_IDLE,""),(2,1.5,C_GEMM,"MoE GEMM"),(3.5,1,C_IDLE,""),
            (4.5,1.5,C_GEMM,"MoE GEMM"),(6,1,C_IDLE,""),(7,1.5,C_GEMM,"MoE GEMM"),(8.5,1.5,C_IDLE,"")])
bars(yA1m, [(0,2.5,C_CD,"A2A dispatch"),(2.5,2.5,C_CC,"A2A combine"),(5,2.5,C_CD,"dispatch"),(7.5,2.5,C_CC,"combine")])
lanelabel(yA1c,"GPU compute"); lanelabel(yA1m,"EP all-to-all"); scen(yA1c,yA1m,"ALPHA  bf16","single-node EP")
wall(10, yA1m-0.15, yA1c+LANE_H+0.15, label="wall-clock")
ax.text(0.1, yA1m-0.45, "comm spans edge-to-edge  =  critical path (sets the wall).  GEMM sits in the idle gaps, in comm's shadow.",
        fontsize=8.3, color=C_CD, style="italic")

# ===== ALPHA fp8 =====
bars(yA2c, [(0,1,C_ATTN,"attn"),(1,1,C_IDLE,""),(2,1.1,C_GEMM,"GEMM↓"),(3.1,0.5,C_CAST,"cast"),
            (3.6,0.9,C_IDLE,""),(4.5,1.1,C_GEMM,"GEMM↓"),(5.6,0.5,C_CAST,"cast"),(6.1,0.9,C_IDLE,""),
            (7,1.1,C_GEMM,"GEMM↓"),(8.1,0.5,C_CAST,"cast"),(8.6,1.4,C_IDLE,"")])
bars(yA2m, [(0,2.5,C_CD,"A2A dispatch"),(2.5,2.5,C_CC,"A2A combine"),(5,2.5,C_CD,"dispatch"),(7.5,2.5,C_CC,"combine")])
lanelabel(yA2c,"GPU compute"); lanelabel(yA2m,"EP all-to-all"); scen(yA2c,yA2m,"ALPHA  fp8","blockwise · H100")
wall(10, yA2m-0.15, yA2c+LANE_H+0.15, color=C_GRAY, dashed=True, lw=1.6)
wall(10.35, yA2m-0.15, yA2c+LANE_H+0.15, label="≈ same (−10.8%)")
ax.text(10.65, (yA2c+yA2m)/2, "comm UNCHANGED\n(fp8 dispatch not in\nthis backend); GEMM\nshrank but was hidden,\ncast tax leaks →\nwall does NOT move",
        fontsize=7.4, color=C_CD, va="center")

# ===== DSV3 bf16 =====
bars(yB1c, [(0,2.5,C_GEMM,"big GEMM"),(2.5,2.5,C_GEMM,"big GEMM"),(5,2.5,C_GEMM,"big GEMM"),(7.5,2.5,C_GEMM,"big GEMM")])
bars(yB1m, [(0,1.2,C_CD,"A2A"),(2.5,1.2,C_CC,"A2A"),(5,1.2,C_CD,"A2A"),(7.5,1.2,C_CC,"A2A")])
lanelabel(yB1c,"GPU compute"); lanelabel(yB1m,"EP all-to-all"); scen(yB1c,yB1m,"DeepSeek-V3  bf16","DualPipe + node-limited")
wall(10, yB1m-0.15, yB1c+LANE_H+0.15, label="wall-clock")
ax.text(0.1, yB1c+LANE_H+0.18, "GEMM packed edge-to-edge  =  critical path (sets the wall).",
        fontsize=8.3, color=C_GEMM, style="italic")
ax.text(0.1, yB1m-0.45, "comm is short (node-limited) and hidden behind compute (DualPipe overlap)  →  NOT on the critical path.",
        fontsize=8.3, color=C_CD, style="italic")

# ===== DSV3 fp8 =====
bars(yB2c, [(0,1.75,C_GEMM,"GEMM↓"),(1.75,0.2,C_CAST,""),(1.95,1.75,C_GEMM,"GEMM↓"),(3.7,0.2,C_CAST,""),
            (3.9,1.75,C_GEMM,"GEMM↓"),(5.65,0.2,C_CAST,""),(5.85,1.75,C_GEMM,"GEMM↓")])
bars(yB2m, [(0,0.9,C_CD,"A2A"),(1.95,0.9,C_CC,"A2A"),(3.9,0.9,C_CD,"A2A"),(5.85,0.9,C_CC,"A2A")])
lanelabel(yB2c,"GPU compute"); lanelabel(yB2m,"EP all-to-all"); scen(yB2c,yB2m,"DeepSeek-V3  fp8","big GEMM + fp8 dispatch")
wall(10, yB2m-0.15, yB2c+LANE_H+0.15, color=C_GRAY, dashed=True, lw=1.6)   # bf16 reference
wall(7.6, yB2m-0.15, yB2c+LANE_H+0.15, color=C_WIN, label="wall ↓  WIN")
ax.annotate("", xy=(7.65, yB2m-0.02), xytext=(9.95, yB2m-0.02),
            arrowprops=dict(arrowstyle="->", color=C_WIN, lw=2.2))
ax.text(8.8, yB2m-0.4, "faster", ha="center", fontsize=9, color=C_WIN, fontweight="bold")

# group headers + separator
ax.text(3.6, 13.55, "ALPHA — communication EXPOSED   (GEMM runs in comm's shadow)",
        ha="center", fontsize=12, fontweight="bold", color="#b91c1c")
ax.axhline(8.05, color="#d1d5db", lw=1)
ax.text(3.6, 7.55, "DeepSeek-V3 — communication HIDDEN   (GEMM is the critical path)",
        ha="center", fontsize=12, fontweight="bold", color="#15803d")

# takeaway
ax.text(-4.9, 2.15, "THE POINT:  fp8 speeds up the GEMM in BOTH cases.  It only shrinks the WALL-CLOCK when the GEMM is on the CRITICAL PATH.",
        fontsize=10.8, fontweight="bold", color="#111827")
ax.text(-4.9, 1.60, "• Alpha (comm exposed): critical path = comm.  fp8 speeds a GEMM that's already hidden → no gain; the cast tax even leaks (−10.8%).",
        fontsize=9.7, color="#374151")
ax.text(-4.9, 1.12, "• DSV3 (comm hidden by DualPipe + node-limited routing): critical path = GEMM → fp8 shrinks the wall → speedup.  Big GEMMs help fp8 more.",
        fontsize=9.7, color="#374151")
ax.text(-4.9, 0.60, "→ To make fp8 help Alpha: first move comm OFF the critical path (hide it), or shrink comm itself (fp8 dispatch — not in this backend).",
        fontsize=9.7, color="#7c3aed", fontweight="bold")

leg = [Patch(fc=C_GEMM,label="GEMM (compute)"), Patch(fc=C_ATTN,label="attention"),
       Patch(fc=C_CD,label="EP all-to-all (comm)"), Patch(fc=C_IDLE,label="GPU idle (waiting)"),
       Patch(fc=C_CAST,label="fp8 cast/quant tax")]
fig.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5,0.935), ncol=5, fontsize=9.5, frameon=False)

ax.text(-4.9, 0.05, "time  →", fontsize=10, color="#4b5563")
ax.set_xlim(-5.1, 13.2); ax.set_ylim(-0.15, 14.0)
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values(): s.set_visible(False)
fig.suptitle("Why fp8 GEMM helps DeepSeek-V3 but not Alpha — it's about the CRITICAL PATH",
             fontsize=14.5, fontweight="bold", y=0.985)
out = "/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/examples/alpha/docs/fp8_critical_path.png"
plt.savefig(out, dpi=145, facecolor="white"); print("wrote", out)

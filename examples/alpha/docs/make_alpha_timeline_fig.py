#!/usr/bin/env python3
"""Alpha real measured timeline — x-axis = wall-clock time (ms), from nsys.
A 1200 ms steady window (analysis_24L, GBS=768, EP=4, rank 0). Shows the actual
compute-busy and comm-busy intervals and the resulting critical-path state, so you
can SEE where the GPU computes, where comm is exposed, and where it overlaps."""
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

d = json.load(open("/tmp/claude-1100/-home-work-vidsearch-repos-project-s-Pai-Megatron-Patch/29ffbc18-6720-47fd-8f11-1153012aeaf6/scratchpad/timeline.json"))
W = d["window_ms"]; st = d["stats"]
CB = "#2563eb"; C_SR="#ea580c"; C_AG="#fb923c"; C_AR="#fdba74"
S_CO="#bfdbfe"; S_OV="#7c3aed"; S_EX="#dc2626"; S_ID="#e5e7eb"

fig, ax = plt.subplots(figsize=(16, 6.6))
BH = 0.7

# ---- lane 1: compute ----
yC = 4.2
ax.broken_barh([(x,w) for x,w in d["compute"]], (yC,BH), facecolors=CB, edgecolor="none")
ax.text(-18, yC+BH/2, "COMPUTE\n(GPU SMs)", ha="right", va="center", fontsize=10, fontweight="bold", color=CB)

# ---- lane 2: communication (by type) ----
yM = 2.9
for iv,c in [(d["sendrecv"],C_SR),(d["allgather"],C_AG),(d["allreduce"],C_AR)]:
    ax.broken_barh([(x,w) for x,w in iv], (yM,BH), facecolors=c, edgecolor="none")
ax.text(-18, yM+BH/2, "COMM\n(EP all-to-all)", ha="right", va="center", fontsize=10, fontweight="bold", color=C_SR)

# ---- ribbon: critical-path state ----
res=0.1; N=int(W/res)
cm=bytearray(N); mm=bytearray(N)
for x,w in d["compute"]:
    for i in range(int(x/res),min(N,int((x+w)/res)+1)): cm[i]=1
for x,w in d["comm_all"]:
    for i in range(int(x/res),min(N,int((x+w)/res)+1)): mm[i]=1
def state(i):
    c,m=cm[i],mm[i]
    return ("ov" if (c and m) else "co" if c else "ex" if m else "id")
yS = 1.7; segs=[]; idle_segs=[]; i=0
while i<N:
    s=state(i); j=i
    while j<N and state(j)==s: j+=1
    col={"co":S_CO,"ov":S_OV,"ex":S_EX,"id":S_ID}[s]
    ax.broken_barh([(i*res,(j-i)*res)], (yS,0.55), facecolors=col, edgecolor="none")
    if s=="ex" and (j-i)*res>=4:
        segs.append((i*res+(j-i)*res/2))
    if s=="id" and (j-i)*res>=15:
        idle_segs.append((i*res+(j-i)*res/2,(j-i)*res))
    i=j
if idle_segs:
    xc,wd=max(idle_segs,key=lambda a:a[1])
    ax.annotate(f"nsys host-stall ~{wd:.0f}ms\n(instrumentation artifact — NOT real idle)",
                xy=(xc,yS), xytext=(xc,yS-0.72), ha="center", fontsize=8.3, color="#6b7280",
                arrowprops=dict(arrowstyle="->", color="#9ca3af", lw=1.4))
ax.text(-18, yS+0.28, "critical-path\nstate", ha="right", va="center", fontsize=9.5, fontweight="bold", color="#374151")
# arrows to a couple of exposed regions
for k,xc in enumerate(sorted(segs, key=lambda a:-a)[:3]):
    ax.annotate("comm EXPOSED\n(GPU waits on A2A)", xy=(xc, yS+0.55), xytext=(xc, yS+1.05),
                ha="center", fontsize=8.2, color=S_EX, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=S_EX, lw=1.6))

# stats bar (right) + labels
tot=W
ax.text(-18, 0.55,
        f"in this {W:.0f} ms window (measured):", ha="right", va="center", fontsize=9, fontweight="bold", color="#111827")
lab=[("compute-only",st["compute_only"],S_CO,"#1e3a8a"),("overlap (both)",st["overlap"],S_OV,"white"),
     ("comm EXPOSED",st["comm_exposed"],S_EX,"white"),("idle (nsys)",st["idle"],S_ID,"#374151")]
x=0
for name,ms,c,tc in lab:
    w=ms/tot*W
    ax.broken_barh([(x,w)],(0.25,0.55),facecolors=c,edgecolor="white")
    ax.text(x+w/2,0.52,f"{name}\n{ms:.0f}ms · {100*ms/tot:.0f}%",ha="center",va="center",fontsize=8,color=tc)
    x+=w

ax.set_xlim(-190, W+8); ax.set_ylim(-0.1, 6.1)
ax.set_yticks([])
ax.set_xlabel("wall-clock time (ms) — measured", fontsize=11)
ax.set_xticks(range(0,int(W)+1,100))
for s in ax.spines.values(): s.set_visible(False)
ax.spines["bottom"].set_visible(True); ax.spines["bottom"].set_color("#9ca3af")
ax.tick_params(axis="x", labelsize=8.5)

leg=[Patch(fc=CB,label="compute (GEMM/Mamba/attn/elementwise)"),
     Patch(fc=C_SR,label="SendRecv (EP all-to-all)"),Patch(fc=C_AG,label="AllGather"),Patch(fc=C_AR,label="AllReduce"),
     Patch(fc=S_EX,label="comm EXPOSED (compute idle → wall spent on comm)")]
ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5,1.16), ncol=5, fontsize=8.3, frameon=False)

ax.text((W)/2, -0.02+5.55, "", ha="center")  # spacer
fig.suptitle("Alpha real timeline — compute vs communication over wall-clock time  (nsys measured · GBS=768 · EP=4 · 1.2 s steady window)",
             fontsize=13, fontweight="bold", y=0.99)
ax.text(W/2, 5.55, "Compute nearly fills the lane (72% busy). Comm (EP A2A) fires in bursts: most overlaps compute, "
        "but 16% is EXPOSED (red) — GPU stalls on the all-to-all. Optimize compute (biggest) AND the red exposed-comm.",
        ha="center", fontsize=9.3, color="#374151", style="italic")
plt.tight_layout()
out="/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/examples/alpha/docs/alpha_timeline.png"
plt.savefig(out, dpi=150, facecolor="white", bbox_inches="tight"); print("wrote", out)

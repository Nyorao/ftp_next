"""Editable, flat-vector rendering of the verified current ContextUNet.

The separate design_reference.png is an image_gen concept. This script authors
the final vector independently, keeping labels and graph connections exact.
"""
from pathlib import Path
import math
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/context-architecture-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, PathPatch
from matplotlib.path import Path as MPath

OUT = Path(__file__).resolve().parent
W, H = 2000, 1320
plt.rcParams.update({
    "font.family": "Liberation Sans", "font.size": 16,
    "mathtext.fontset": "dejavusans", "svg.fonttype": "none",
    "pdf.fonttype": 42, "savefig.facecolor": "white",
    "axes.unicode_minus": False,
})
fig = plt.figure(figsize=(20, 13.2), dpi=100, facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.axis("off")
C = {
    "ink": "#344557", "line": "#7F90A0", "outline": "#B9C8D5",
    "panel": "#D8E1E9", "blue": "#EAF1F9", "blue2": "#D6E5F3",
    "blue_edge": "#96B1CA", "green": "#E7F2EC",
    "green_edge": "#A1B9AC", "purple": "#EEEAF7",
    "purple_edge": "#B2A5CD", "orange": "#FCF1E2",
    "orange_edge": "#D6BE9C", "gray": "#F3F6F9", "white": "#FFFFFF",
}

def txt(x, y, value, size=25, weight="normal", ha="center", color=None, **kw):
    return ax.text(x, y, value, color=color or C["ink"], fontsize=size*.72,
                   ha=ha, va="center", weight=weight, linespacing=1.22,
                   zorder=8, **kw)

def box(x, y, w, h, fill="white", edge="outline", radius=10, lw=1.25, z=3):
    patch = FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=C.get(fill, fill), edgecolor=C.get(edge, edge),
        linewidth=lw, zorder=z)
    ax.add_patch(patch)
    return patch

def module(x, cy, w, h, label, fill="blue", edge=None, size=25, weight="normal"):
    box(x, cy-h/2, w, h, fill=fill,
        edge=edge or {"blue": "blue_edge", "green": "green_edge",
                      "purple": "purple_edge", "orange": "orange_edge"}.get(fill, "outline"))
    txt(x+w/2, cy, label, size=size, weight=weight)

def line(points, color="line", lw=1.6, dashed=False, z=4):
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=C.get(color, color), linewidth=lw,
            linestyle=(0,(4,3)) if dashed else "-", solid_capstyle="round",
            solid_joinstyle="round", zorder=z)

def arrow(points, color="line", lw=1.65, dashed=False, head=13):
    if len(points) > 2:
        line(points[:-1], color=color, lw=lw, dashed=dashed)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
        mutation_scale=head, color=C.get(color,color), linewidth=lw,
        linestyle=(0,(4,3)) if dashed else "-", shrinkA=0, shrinkB=0,
        joinstyle="round", zorder=5))

def dot(x, y, r=3, color="line"):
    ax.add_patch(Circle((x,y), r, facecolor=C.get(color,color), edgecolor="none", zorder=6))

def panel(x, y, w, h, label, title):
    box(x,y,w,h,edge="panel",radius=17,lw=1.15,z=0)
    txt(x+26,y+34,f"({label})  {title}",size=31,weight="bold",ha="left")

def spine(cx, cy, width=45, height=90, segmented=False):
    """Abstract sagittal anatomy; no actual patient image or model prediction."""
    top = cy-height/2
    dy = height/7
    for i in range(7):
        offset = 7*math.sin((i/6)*math.pi-.45)
        fill = ("purple" if i%2==0 else "blue2") if segmented else "gray"
        box(cx-width*.42+offset, top+i*dy, width*.64,dy*.73,
            fill=fill,edge="purple_edge" if segmented else "line",radius=3,lw=1)
        if segmented and i<6:
            box(cx-width*.4+offset,top+i*dy+dy*.76,width*.58,dy*.17,
                fill="green",edge="green_edge",radius=1.3,lw=.65)
    p = [(cx+width*.54+7*math.sin((i/24)*math.pi-.45),top+i*height/24)
         for i in range(25)]
    line(p,color="green_edge" if segmented else "line",lw=2.1)

def tiles(x, cy, w=46, h=32):
    for i, fill in enumerate(("blue", "blue2", "blue")):
        box(x+i*(w+12),cy-h/2,w,h,fill=fill,edge="blue_edge",radius=4,lw=.8)
        line([(x+i*(w+12)+7,cy-6),(x+i*(w+12)+w-7,cy-6)],
             color="blue_edge",lw=.6,z=5)
        line([(x+i*(w+12)+7,cy+3),(x+i*(w+12)+w-7,cy+3)],
             color="blue_edge",lw=.6,z=5)

# (a) Exact five-level backbone and four-level decoder.
panel(20,15,1960,610,"a","Three-slice context network")
txt(190,127,"Adjacent sagittal slices",size=27,weight="bold")
for cy,label,fill in [(230,r"$x_{z-1}$","gray"),
                      (380,r"$x_z$","blue"),
                      (530,r"$x_{z+1}$","gray")]:
    box(65,cy-54,250,108,fill=fill,edge="blue_edge",radius=11,
        lw=1.65 if cy==380 else 1)
    spine(126,cy,width=42,height=82)
    txt(239,cy,label,size=30)
    line([(315,cy),(350,cy)])
line([(350,230),(350,530)])
dot(350,230)
arrow([(350,230),(425,230)])

box(395,95,555,506,fill="white",edge="blue_edge",radius=14,lw=1.1)
txt(672,133,"Shared 2D encoder",size=30,weight="bold")
txt(672,169,"Tied weights across slices",size=23)
rows=[230,312,394,476,558]
res=[224,112,56,28,14]
channels=[32,64,128,256,256]
for level,(cy,spatial,ch) in enumerate(zip(rows,res,channels)):
    box(425,cy-27,495,54,fill="blue",edge="blue_edge",radius=8,lw=.9)
    txt(446,cy,f"E{level}  |  {spatial} × {spatial}  |  {ch}",size=24,ha="left")
    tiles(743,cy,w=45,h=31)
    if level<4:
        arrow([(675,cy+27),(675,rows[level+1]-27)],lw=1.4,head=11)
    if level<2:
        arrow([(920,cy),(1000,cy)])
        module(1000,cy,180,48,"Center only",size=25)
        arrow([(1180,cy),(1360,cy)])
    else:
        arrow([(920,cy),(1000,cy)])
        module(1000,cy,180,48,f"CRF{level}",fill="green",size=26,weight="bold")
        if level<4:
            arrow([(1180,cy),(1360,cy)])
        else:
            arrow([(1180,cy),(1445,cy),(1445,503)])

box(1300,110,300,487,fill="white",edge="purple_edge",radius=14,lw=1.1)
txt(1450,148,"U-Net decoder",size=29,weight="bold")
txt(1450,181,"Four upsampling stages",size=22)
for i,cy in enumerate(rows[:4]):
    module(1360,cy,170,54,f"D{i}",fill="purple",size=28,weight="bold")
    if i<3:
        arrow([(1445,rows[i+1]-27),(1445,cy+27)])
arrow([(1530,230),(1615,230),(1615,340),(1640,340)])
module(1640,340,130,66,"1 × 1\nhead",size=25)
arrow([(1770,340),(1795,340)])
box(1795,200,165,280,fill="white",edge="purple_edge",radius=12)
txt(1877,232,"Center mask",size=23,weight="bold")
txt(1877,264,"20 classes",size=23)
spine(1874,365,width=62,height=139,segmented=True)

# (b) Spatial weighting of all three inputs, followed by center residual.
panel(20,650,800,475,"b","Center-residual fusion")
txt(756,686,r"$\mathrm{CRF}_k$",size=27)
for cy,label in [(782,r"$F_{-}$"),(865,r"$F_0$"),(948,r"$F_{+}$")]:
    module(55,cy,85,48,label,size=28)
    line([(140,cy),(173,cy)])
    dot(173,cy,r=2.5)
line([(173,782),(173,948)])
arrow([(173,782),(210,782)])
module(210,782,130,54,"Concat",size=25)
arrow([(340,782),(380,782)])
module(380,782,135,54,"1 × 1 Conv",size=23)
arrow([(515,782),(555,782)])
module(555,782,135,54,"Softmax",size=24)
arrow([(622,809),(622,854),(393,854),(393,922)])
txt(466,831,r"$w_{-},\ w_0,\ w_{+}$",size=26)
txt(200,892,"Per-pixel weights",size=20,ha="left")
arrow([(173,948),(173,955),(310,955)])
module(310,955,165,66,"Weighted\nsum",fill="green",size=25)
arrow([(475,955),(550,955)])
txt(512,931,r"$F_w$",size=25)
module(550,955,150,66,"Center\nresidual",fill="green",size=25)
module(700,850,62,36,r"$F_0$",size=25)
arrow([(731,868),(731,900),(625,900),(625,922)])
arrow([(700,955),(764,955)])
txt(779,955,r"$\widetilde{F}$",size=31)
txt(625,1018,r"$\gamma=0$ at initialization",size=20)
txt(420,1082,r"$\widetilde{F}=F_0+\gamma\,(F_w-F_0)$",size=30)

# (c) Training: single trainable student, its frozen EMA, aligned pseudo targets.
panel(850,650,1130,475,"c","Mean Teacher training")
module(885,810,180,78,"Labeled\ntriplets",size=26)
module(885,985,180,78,"Unlabeled\ntriplets",size=26)
arrow([(1065,810),(1135,810)])
module(1135,810,160,68,"Strong\nview",size=26)
arrow([(1295,810),(1360,810)])
line([(1065,985),(1100,985)])
dot(1100,985)
arrow([(1100,985),(1100,915),(1135,915)])
module(1135,915,160,68,"Strong view\n+ CutMix",size=24)
arrow([(1295,915),(1360,915)])
arrow([(1100,985),(1100,1030),(1135,1030)])
module(1135,1030,160,68,"Original\nview",size=25)
arrow([(1295,1030),(1360,1030)])
box(1360,775,190,155,fill="purple",edge="purple_edge",radius=12)
txt(1455,820,"Student",size=29,weight="bold")
txt(1455,860,"ContextUNet",size=25)
txt(1455,899,"Trainable",size=22)
box(1360,984,190,92,fill="gray",edge="outline",radius=12)
txt(1455,1011,"EMA teacher",size=27,weight="bold")
txt(1455,1049,"Stop gradient",size=22)
arrow([(1455,930),(1455,984)],dashed=True,head=12)
txt(1500,958,"EMA",size=21,ha="left")

arrow([(1550,810),(1670,810)])
txt(1610,786,r"$p_l$",size=25)
module(1670,810,270,66,"0.5 CE + 0.5 Dice",fill="orange",size=25)
txt(1805,745,r"$y_z$  (center GT)",size=23)
arrow([(1805,760),(1805,777)],head=10)
arrow([(1550,895),(1622,895),(1622,915),(1670,915)])
txt(1605,872,r"$p_u$",size=25)
module(1670,915,270,66,"Masked CE",fill="orange",size=27)
arrow([(1550,1030),(1585,1030)])
module(1585,1030,155,66,"Confidence\nmask",size=24)
arrow([(1740,1030),(1785,1030)])
module(1785,1030,155,66,"Aligned\nCutMix",size=24)
arrow([(1862,997),(1862,948)])
txt(1894,975,r"$\hat{y}_u$",size=25)
txt(1530,1095,r"$\mathcal{L}=\mathcal{L}_{\mathrm{sup}}+\lambda(t)\,\mathcal{L}_{u}$",size=30)

# (d) Inference-only feature reuse; one network is deployed.
panel(20,1155,1960,145,"d","Cached volume inference")
module(45,1251,285,42,"Single EMA network",size=24)
box(395,1215,225,65,fill="white",edge="outline",radius=9)
for dx in (0,6,12):
    box(414+dx,1225+dx/3,35,43,fill="white",edge="blue_edge",radius=3,lw=.8)
txt(533,1248,"Volume",size=26)
arrow([(620,1248),(690,1248)])
module(690,1248,205,65,"Encode once",size=26)
arrow([(895,1248),(965,1248)])
module(965,1248,230,65,"Feature cache",fill="green",size=26)
arrow([(1195,1248),(1265,1248)])
module(1265,1248,230,65,"Fuse + decode",fill="purple",size=26)
arrow([(1495,1248),(1565,1248)])
box(1565,1215,375,65,fill="white",edge="purple_edge",radius=9)
spine(1607,1248,width=24,height=48,segmented=True)
txt(1788,1248,"3D segmentation",size=27)

stem=OUT/"context_architecture"
for ext in ("svg","pdf"):
    fig.savefig(stem.with_suffix("."+ext),format=ext,facecolor="white")
fig.savefig(stem.with_suffix(".png"),dpi=200,facecolor="white")
fig.savefig(OUT/"preview.png",dpi=100,facecolor="white")
print("Saved context_architecture.svg, .pdf and .png (4000 × 2640)")

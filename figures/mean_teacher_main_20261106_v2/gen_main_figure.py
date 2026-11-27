"""Editable Mean Teacher main figure, with two architecture insets.

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
W, H = 2100, 1350
plt.rcParams.update({
    "font.family": "Liberation Sans", "font.size": 16,
    "mathtext.fontset": "dejavusans", "svg.fonttype": "none",
    "pdf.fonttype": 42, "savefig.facecolor": "white",
    "axes.unicode_minus": False,
})
fig = plt.figure(figsize=(21, 13.5), dpi=100, facecolor="white")
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

def triplet_card(x, cy, title, symbol):
    box(x,cy-83,225,166,fill="white",edge="outline",radius=12)
    txt(x+112.5,cy-58,title,size=25,weight="bold")
    for i in range(3):
        px=x+18+i*64
        box(px,cy-37,56,76,fill="blue" if i==1 else "gray",
            edge="blue_edge",radius=5,lw=1.2 if i==1 else .8)
        spine(px+27,cy+1,width=20,height=56)
    txt(x+112.5,cy+61,symbol,size=25)

def prediction(x, cy, symbol):
    box(x,cy-34,145,68,fill="white",edge="purple_edge",radius=9)
    spine(x+30,cy,width=21,height=46,segmented=True)
    txt(x+96,cy,symbol,size=29)

def network_group(y, center_y, teacher=False):
    box(670,y,810,230 if not teacher else 215,fill="white",
        edge="outline" if teacher else "blue_edge",radius=15,lw=1.6)
    if teacher:
        txt(705,y+37,r"EMA teacher  $f_{\bar{\theta}}$",size=33,weight="bold",ha="left")
        module(1272,y+37,175,36,"Stop gradient",fill="gray",size=21)
    else:
        txt(705,y+37,r"Student  $f_\theta$",size=34,weight="bold",ha="left")
        module(1297,y+37,150,36,"Trainable",fill="purple",size=22)
    # Center features bypass deep fusion at the two shallow encoder scales.
    bypass_y=center_y-58
    arrow([(802,center_y-40),(802,bypass_y),(1330,bypass_y),(1330,center_y-40)],
          color="blue_edge",lw=1.35,head=10)
    txt(1065,bypass_y-17,"Center skips",size=21,color=C["blue_edge"])
    module(710,center_y,185,80,"Shared 2D\nencoder",size=27)
    arrow([(895,center_y),(960,center_y)])
    module(960,center_y,195,80,"Multi-scale\nCRF (c)",fill="green",size=28)
    arrow([(1155,center_y),(1220,center_y)])
    module(1220,center_y,220,80,"Decoder\n+ head",fill="purple",size=28)

# (a) The Mean Teacher system is the dominant visual subject.
panel(20,20,2060,830,"a","Mean Teacher framework")
txt(2039,58,"Center-slice segmentation",size=24,ha="right")
triplet_card(50,260,"Labeled triplets",r"$x_l$")
triplet_card(50,625,"Unlabeled triplets",r"$x_u$")
arrow([(275,260),(350,260)])
module(350,260,205,76,"Strong view",size=27)
line([(555,260),(620,260),(620,350)])
line([(275,625),(305,625)])
dot(305,625)
arrow([(305,625),(305,415),(350,415)])
module(350,415,205,90,"Strong view\nCutMix M",size=27)
line([(555,415),(620,415),(620,350)])
dot(620,350)
arrow([(620,350),(710,350)])
arrow([(305,625),(305,690),(350,690)])
module(350,690,205,76,"Original view",size=26)
arrow([(555,690),(710,690)])
network_group(215,350,teacher=False)
network_group(565,690,teacher=True)
arrow([(1065,445),(1065,565)],dashed=True,head=14)
txt(1110,489,"EMA update",size=26,ha="left")
txt(1110,526,r"$\bar{\theta}\leftarrow\mathrm{EMA}(\theta)$",size=24,ha="left")

# Student produces labeled and unlabeled predictions from the same network.
line([(1440,350),(1515,350)])
dot(1515,350)
arrow([(1515,350),(1515,280),(1555,280)])
arrow([(1515,350),(1515,420),(1555,420)])
prediction(1555,280,r"$p_l^{s}$")
prediction(1555,420,r"$p_u^{s}$")
arrow([(1700,280),(1775,280)])
module(1775,280,270,82,"Supervised loss\n0.5 CE + 0.5 Dice",fill="orange",size=25)
module(1800,181,220,46,r"$y_l$  ·  Center GT",fill="white",size=25)
arrow([(1910,204),(1910,239)])
arrow([(1700,420),(1740,420),(1740,455),(1775,455)])
module(1775,455,270,82,"Unsupervised loss\nMasked CE",fill="orange",size=26)

# Actual target order: probabilities -> same CutMix M -> confidence -> loss.
arrow([(1440,690),(1555,690)])
prediction(1555,690,r"$p_u^{t}$")
arrow([(1700,690),(1775,690)])
module(1775,690,270,70,"Target CutMix M",size=27)
arrow([(1910,655),(1910,605)])
module(1775,570,270,70,"Confidence filter",size=27)
arrow([(1910,535),(1910,496)])
txt(1941,516,r"$\hat{y}_u$",size=28,ha="left")
module(50,797,375,42,"M: shared CutMix mask",fill="white",size=24)
txt(1075,817,r"$\mathcal{L}=\mathcal{L}_{\mathrm{sup}}+\lambda(t)\,\mathcal{L}_{u}$",size=31)
module(1755,797,290,42,"Inference: teacher only",fill="gray",size=24)

# (b) Each complete student/teacher contains this identical architecture.
panel(20,885,1280,445,"b","Three-slice ContextUNet")
txt(1265,920,"Student & teacher",size=23,ha="right")
txt(165,1000,"Adjacent slices",size=25,weight="bold")
for i,label in enumerate((r"$x_{z-1}$",r"$x_z$",r"$x_{z+1}$")):
    px=58+i*73
    box(px,1050,61,96,fill="blue" if i==1 else "white",edge="blue_edge",radius=5)
    spine(px+29,1095,width=24,height=69)
    txt(px+30,1173,label,size=24)
line([(88,1050),(88,1030),(234,1030),(234,1050)])
line([(161,1050),(161,1030)])
arrow([(234,1030),(280,1030),(280,1010),(340,1010)])
txt(163,1251,"Shared weights",size=23)
txt(480,967,"Shared encoder",size=27,weight="bold")
txt(1080,967,"Decoder",size=26,weight="bold")
ys=[1010,1068,1126,1184,1242]
sizes=[224,112,56,28,14]
for k,(cy,spatial) in enumerate(zip(ys,sizes)):
    box(340,cy-19,280,38,fill="blue",edge="blue_edge",radius=6,lw=.85)
    txt(357,cy,f"E{k} · {spatial}²",size=23,ha="left")
    for j in range(3):
        box(528+j*27,cy-10,21,20,fill="blue2" if j==1 else "white",
            edge="blue_edge",radius=2,lw=.7)
    if k<4:
        arrow([(480,cy+19),(480,ys[k+1]-19)],head=10,lw=1.15)
    arrow([(620,cy),(700,cy)],head=11,lw=1.4)
    module(700,cy,165,38,"Center only" if k<2 else f"CRF{k}",
           fill="blue" if k<2 else "green",size=23,
           weight="normal" if k<2 else "bold")
    if k<4:
        arrow([(865,cy),(1030,cy)],head=11,lw=1.4)
    else:
        arrow([(865,cy),(1080,cy),(1080,1203)],head=11,lw=1.4)
for i,cy in enumerate(ys[:4]):
    module(1030,cy,100,38,f"D{i}",fill="purple",size=24,weight="bold")
    if i<3:
        arrow([(1080,ys[i+1]-19),(1080,cy+19)],head=10,lw=1.3)
arrow([(1130,1010),(1180,1010)],head=11,lw=1.4)
module(1180,1010,90,50,"1 × 1\nhead",size=21)
arrow([(1225,1035),(1225,1080)],head=10,lw=1.3)
box(1180,1080,90,134,fill="white",edge="purple_edge",radius=7,lw=.9)
spine(1224,1147,width=39,height=106,segmented=True)
txt(1225,1255,"20-class\ncenter mask",size=22)
txt(480,1291,"Five feature scales",size=22)
txt(790,1291,"Deep-scale fusion",size=22)

# (c) Exact fusion: three inputs to Concat AND to the weighted sum.
panel(1335,885,745,445,"c","Center-residual fusion")
txt(2046,920,r"$\mathrm{CRF}_k$",size=26,ha="right")
for cy,label in [(1005,r"$F_{-}$"),(1078,r"$F_0$"),(1151,r"$F_{+}$")]:
    module(1360,cy,65,42,label,size=27)
    line([(1425,cy),(1445,cy)],lw=1.35)
    dot(1445,cy,r=2.2)
line([(1445,1005),(1445,1151)],lw=1.35)
arrow([(1445,1005),(1475,1005)],head=10,lw=1.3)
module(1475,1005,110,45,"Concat",size=23)
arrow([(1585,1005),(1620,1005)],head=10,lw=1.3)
module(1620,1005,115,45,"1 × 1 Conv",size=21)
arrow([(1735,1005),(1770,1005)],head=10,lw=1.3)
module(1770,1005,125,45,"Softmax",size=23)
arrow([(1832,1027.5),(1832,1070),(1605,1070),(1605,1140)],head=11,lw=1.4)
txt(1668,1048,r"$w_{-},\ w_0,\ w_{+}$",size=25)
txt(1465,1103,"Spatial weights",size=20,ha="left")
arrow([(1445,1151),(1445,1170),(1530,1170)],head=11,lw=1.4)
module(1530,1170,150,60,"Weighted\nsum",fill="green",size=25)
arrow([(1680,1170),(1800,1170)],head=11,lw=1.4)
txt(1740,1147,r"$F_w$",size=25)
module(1800,1170,155,60,"Center\nresidual",fill="green",size=25)
module(1940,1080,65,36,r"$F_0$",size=25)
arrow([(1972.5,1098),(1972.5,1120),(1877,1120),(1877,1140)],head=10,lw=1.35)
arrow([(1955,1170),(2020,1170)],head=11,lw=1.4)
txt(2040,1170,r"$\widetilde{F}$",size=31)
module(1923,1238,115,35,r"$\gamma_0=0$",fill="white",size=23)
txt(1690,1281,r"$\widetilde{F}=F_0+\gamma\,(F_w-F_0)$",size=29)

stem=OUT/"mean_teacher_main_v2"
for ext in ("svg","pdf"):
    fig.savefig(stem.with_suffix("."+ext),format=ext,facecolor="white")
fig.savefig(stem.with_suffix(".png"),dpi=200,facecolor="white")
fig.savefig(OUT/"preview.png",dpi=100,facecolor="white")
print("Saved mean_teacher_main_v2.svg, .pdf and .png (4200 × 2700)")


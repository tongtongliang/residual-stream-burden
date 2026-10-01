"""Editable SVG and vector PDF diagrams for the manuscript (no TikZ)."""
from pathlib import Path
import math
from reportlab.graphics.shapes import Drawing, Rect, Line, Polygon, String, Circle, Path as DrawingPath
from reportlab.graphics import renderPDF, renderSVG
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Embed fonts in PDFs so rendering does not depend on local font substitution.
import os
if os.environ.get('DIAGRAM_FONT_DIR'):
    FONTROOT=Path(os.environ['DIAGRAM_FONT_DIR'])
else:
    import matplotlib
    FONTROOT=Path(matplotlib.get_data_path())/'fonts/ttf'
pdfmetrics.registerFont(TTFont('DiagramSans',str(FONTROOT/'DejaVuSans.ttf')))
pdfmetrics.registerFont(TTFont('DiagramSansBold',str(FONTROOT/'DejaVuSans-Bold.ttf')))

OUT = Path(__file__).resolve().parents[1] / 'figure/manuscript_panels'
INK='#273442'; BLUE='#D9E8F7'; GREEN='#DDEDDC'; ORANGE='#F9E4CB'; PURPLE='#E9DFF1'
COLORS=[BLUE,GREEN,ORANGE,PURPLE]

class Panel:
    def __init__(self,w,h): self.w=w; self.h=h; self.d=Drawing(w,h)
    def text(self,x,y,text,size=8,bold=False,anchor='middle',color=INK):
        for i,line in enumerate(text.split('\n')):
            self.d.add(String(x,self.h-y-i*size*1.12,line,fontName='DiagramSansBold' if bold else 'DiagramSans',fontSize=size,textAnchor=anchor,fillColor=HexColor(color)))
    def box(self,x,y,w,h,text='',fill=BLUE,size=8,stroke=INK,r=3):
        self.d.add(Rect(x,self.h-y-h,w,h,rx=r,ry=r,fillColor=HexColor(fill),strokeColor=HexColor(stroke),strokeWidth=.65))
        if text:
            width=max(pdfmetrics.stringWidth(t,'DiagramSans',size) for t in text.split('\n'))
            size*=min(1,(w-6)/width)
            self.text(x+w/2,y+h/2+size*.35-(len(text.split('\n'))-1)*size*.56,text,size)
    def line(self,points,arrow=True,color=INK,width=.8,dash=False):
        for (x,y),(a,b) in zip(points,points[1:]):
            self.d.add(Line(x,self.h-y,a,self.h-b,strokeColor=HexColor(color),strokeWidth=width,strokeDashArray=[2,2] if dash else None))
        if arrow:
            (x,y),(a,b)=points[-2:]; ang=math.atan2(b-y,a-x); s=3
            ps=[a,self.h-b,a-s*math.cos(ang)+s*.45*math.sin(ang),self.h-(b-s*math.sin(ang)-s*.45*math.cos(ang)),a-s*math.cos(ang)-s*.45*math.sin(ang),self.h-(b-s*math.sin(ang)+s*.45*math.cos(ang))]
            self.d.add(Polygon(ps,fillColor=HexColor(color),strokeColor=None))
    def plus(self,x,y,r=4):
        self.d.add(Circle(x,self.h-y,r,fillColor=HexColor('#FFFFFF'),strokeColor=HexColor(INK),strokeWidth=.7))
        self.line([(x-2,y),(x+2,y)],False);self.line([(x,y-2),(x,y+2)],False)
    def slots(self,x,y,w=29,h=6,gap=2):
        for i,c in enumerate(COLORS):self.box(x,y+i*(h+gap),w,h,fill=c,r=1)
    def grid(self,x,y,s=23):
        for i,c in enumerate(COLORS):self.box(x+(i%2)*s/2,y+(i//2)*s/2,s/2,s/2,fill=c,r=0)
    def save(self,name):
        OUT.mkdir(parents=True,exist_ok=True)
        renderPDF.drawToFile(self.d,str(OUT/(name+'.pdf')))
        renderSVG.drawToFile(self.d,str(OUT/(name+'.svg')))
        # SVG viewers need public font names rather than ReportLab's aliases.
        svg=OUT/(name+'.svg')
        svg.write_text(svg.read_text().replace('font-family: DiagramSansBold',
            'font-family: DejaVu Sans, Arial, sans-serif; font-weight: bold')
            .replace('font-family: DiagramSans',
            'font-family: DejaVu Sans, Arial, sans-serif'))

def taxonomy():
    p=Panel(132,100)
    p.text(29,8,'Noisy input',6.8)
    p.text(103,8,'Predictions',6.8)
    p.line([(29,12),(29,21)]);p.line([(103,21),(103,12)])
    # Wide -> square -> tall blocks, mirrored along a U-shaped path.
    stages=[(26,44,10),(48,20,20),(79,10,28)]
    for i,(cy,w,h) in enumerate(stages):
        for cx,fill in [(29,BLUE),(103,GREEN)]:
            p.box(cx-w/2,cy-h/2,w,h,fill=fill,r=1.4)
        if i<2:
            p.line([(29+w/2,cy),(103-w/2,cy)],dash=True,width=.65)
        if i<len(stages)-1:
            ny,nw,nh=stages[i+1]
            p.line([(29,cy+h/2),(29,ny-nh/2)])
            p.line([(103,ny-nh/2),(103,cy+h/2)])
    p.text(66,21,'skips',6.2)
    p.line([(34,66),(98,66)],dash=True,width=.65)
    p.box(42,70,48,18,'Conv / ViT\nblocks',BLUE,7)
    p.line([(34,79),(42,79)]);p.line([(90,79),(98,79)])
    p.save('architecture_hierarchical')
    for kind in ['terminal', 'persistent']:
        p=Panel(150,100)
        p.text(75,8,'Noisy input',7)
        p.line([(75,12),(26,12),(26,20)])
        p.line([(75,12),(123,12),(123,66 if kind=='terminal' else 20)])
        p.box(4,20,44,14,'Large patch',BLUE,7)
        p.box(4,44,44,40,'Plain\nDiTs',BLUE,8)
        p.line([(26,34),(26,44)])
        if kind=='terminal':
            p.box(101,66,44,18,'Output\nmodule',GREEN,7)
            p.line([(48,75),(101,75)])
            p.text(80,69,'condition',5.8)
        else:
            p.box(101,20,44,14,'Small patch',GREEN,7)
            p.line([(123,34),(123,44)])
            for y in [44,68]:
                p.box(101,y,44,16,'DiT blocks',GREEN,7)
                p.line([(48,y+8),(101,y+8)])
                p.text(81,y+3,'Cross-attn.',5.5)
            p.line([(123,60),(123,68)])
        p.line([(123,84),(123,91)]);p.text(123,99,'Predictions',6.8)
        p.save('architecture_'+kind)

def hc():
    p=Panel(408,112)
    p.text(75,10,'Plain residual block',9,True)
    p.text(286,10,'Hyper-Connection',9,True)
    p.box(3,24,145,78,fill='#F6F8FA',stroke='#F6F8FA',r=6)
    p.box(166,24,239,78,fill='#F1F6FA',stroke='#F1F6FA',r=6)
    p.box(12,69,28,8,fill=BLUE,r=1);p.box(111,69,28,8,fill=BLUE,r=1)
    p.box(47,35,49,19,'MLP',ORANGE,8)
    p.line([(40,73),(105,73)]);p.plus(105,73);p.line([(109,73),(111,73)])
    p.line([(35,69),(35,44),(47,44)]);p.line([(96,44),(105,44),(105,69)])
    p.text(76,92,'One retained vector',7)
    p.slots(173,56,25,5,2);p.slots(372,56,25,5,2)
    p.box(215,34,33,17,'Read',GREEN,7.5)
    p.box(264,34,40,17,'MLP',ORANGE,8)
    p.box(322,34,33,17,'Write',GREEN,7.5)
    for y in [58.5,65.5,72.5,79.5]:p.line([(198,y),(203,y)],False,width=.6)
    p.line([(203,58.5),(203,79.5)],False,width=.6)
    p.line([(203,63),(207,63),(207,42),(215,42)])
    p.line([(248,42),(264,42)]);p.line([(304,42),(322,42)])
    p.line([(355,42),(363,42),(363,64)]);p.plus(363,68)
    p.line([(203,76),(207,76),(207,87),(363,87),(363,72)])
    p.line([(367,68),(369,68)],False)
    p.line([(369,58.5),(369,79.5)],False,width=.6)
    for y in [58.5,65.5,72.5,79.5]:p.line([(369,y),(372,y)],True,width=.6)
    p.text(285,63,'One workspace',7)
    p.text(285,99,'Carry streams forward',7)
    p.save('hc_read_compute_write')

def sihc():
    p=Panel(408,168)
    p.box(0,0,408,66,fill='#F1F6FA',stroke='#F1F6FA',r=5)
    p.text(7,12,'Spatially indexed state, shared Transformer workspace',9,True,anchor='start')
    p.grid(9,25,25);p.text(22,62,'P8 patches',6.6)
    p.box(48,28,43,22,'Linear\nembed',fill='#FFFFFF',size=7)
    p.slots(107,23,29,6,2)
    p.box(157,27,93,25,'Read → DiT → Write'.replace('→','/'),ORANGE,8)
    p.slots(271,23,29,6,2)
    p.box(317,28,43,22,'Linear\nhead',fill='#FFFFFF',size=7)
    p.grid(375,25,25);p.text(387,62,'Pixels',6.6)
    for a,b in [(34,48),(91,107),(136,157),(250,271),(300,317),(360,375)]:p.line([(a,39),(b,39)])
    p.text(203,61,'Repeat across depth',7)
    p.text(7,81,'Inside one block',9,True,anchor='start')
    p.box(1,89,98,65,fill='#F6F8FA',stroke='#F6F8FA',r=5)
    p.text(50,101,'1  Read each channel',7.5,True)
    for j,c in enumerate(COLORS):
        p.box(9+j*19,112,15,7,fill=c,r=1)
        p.line([(16+j*19,119),(50,130)],width=.6)
    p.box(37,130,26,7,fill=BLUE,r=1)
    p.text(50,149,'Mix slots channel by channel',6.5)
    p.box(117,89,171,65,fill='#FFF3E5',stroke='#FFF3E5',r=5)
    p.text(202,101,'2  Compute in the workspace',7.5,True)
    p.box(128,113,69,22,'Attention',BLUE,8)
    p.box(210,113,66,22,'MLP',BLUE,8)
    p.line([(197,124),(210,124)])
    p.text(202,149,'256 visual tokens; ordinary DiT width',7)
    p.box(306,89,102,65,fill='#F6F8FA',stroke='#F6F8FA',r=5)
    p.text(357,101,'3  Write each channel',7.5,True)
    p.box(344,110,26,7,fill=ORANGE,r=1)
    for j,c in enumerate(COLORS):
        p.line([(357,117),(321+j*23,129)],width=.6)
        p.box(314+j*23,131,15,7,fill=c,r=1)
    p.text(357,149,'Add to the corresponding slots',6.4)
    p.line([(99,123),(117,123)]);p.line([(288,123),(306,123)])
    p.text(204,165,'Four P8 slots per P16 cell; P4 uses sixteen slots. Slot locations persist across all blocks.',7)
    p.save('sihc_spatial_workspace')

if __name__=='__main__':
    taxonomy();hc();sihc();print('Wrote five editable SVG/PDF architecture panels.')

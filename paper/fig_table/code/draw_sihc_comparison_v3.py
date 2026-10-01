"""Two-stream comparison with explicit SVG subscripts and tensor dimensions."""
from pathlib import Path
from html import escape

OUT=Path(__file__).resolve().parents[1]/'figure/algorithm_drafts'
INK='#26364A'; LINE='#89939F'; BLUE='#E4EFF9'; ORANGE='#F8E7D4'; GREEN='#E8F0E3'; GRAY='#ECEFF2'
p=['<svg xmlns="http://www.w3.org/2000/svg" width="1240" height="746" viewBox="0 0 1240 746">',
   '<rect width="1240" height="746" fill="white"/>',
   '<defs><marker id="a" markerWidth="6" markerHeight="6" refX="5.4" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="#89939F"/></marker></defs>']
def txt(x,y,t,size=15,anchor='middle',math=False):
    font='Georgia, Times New Roman, serif' if math else 'Arial, Helvetica, sans-serif'
    p.append(f'<text x="{x}" y="{y}" font-family="{font}" font-size="{size}" text-anchor="{anchor}" fill="{INK}">{t if math else escape(t)}</text>')
def sub(v,s):return f'<tspan font-style="italic">{v}</tspan><tspan dy="5" font-size="12">{s}</tspan><tspan dy="-5">&#8203;</tspan>'
def shape(dim):return f' ∈ ℝ<tspan dy="-7" font-size="12">{dim}</tspan><tspan dy="7">&#8203;</tspan>'
def box(x,y,w,h,fill='white',r=4):p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{LINE}" stroke-width="1.2"/>')
def line(pts,arrow=True,dash=False):
    d='M'+' L'.join(f'{x},{y}' for x,y in pts)
    p.append(f'<path d="{d}" fill="none" stroke="{LINE}" stroke-width="1.5"'+(' stroke-dasharray="4 4"' if dash else '')+(' marker-end="url(#a)"' if arrow else '')+'/>')
def plus(x,y):
    p.append(f'<circle cx="{x}" cy="{y}" r="11" fill="white" stroke="{LINE}" stroke-width="1.5"/>')
    line([(x-6,y),(x+6,y)],False);line([(x,y-6),(x,y+6)],False)
def states(cx,y,n=2,spatial=False,label='ℓ',dims=True):
    if n==1:box(cx-52,y,104,18,GRAY)
    else:
        box(cx-66,y,62,18,BLUE if spatial else GRAY)
        box(cx+4,y,62,18,ORANGE if spatial else GRAY)
    txt(cx,y-10,sub('h',label)+(shape('1×c' if n==1 else '2×c') if dims else ''),17,math=True)
def mat(cx,y,var,dim):
    box(cx-64,y,128,36,GREEN)
    txt(cx,y+23,sub(var,'ℓ')+shape(dim),17,math=True)
def feature(cx,y,name):
    box(cx-24,y,48,13,GRAY)
    txt(cx+36,y+12,sub(name,'ℓ')+shape('1×c'),15,anchor='start',math=True)
def patch(cx,y,spatial=False):
    if spatial:
        p.append(f'<rect x="{cx-66}" y="{y}" width="66" height="22" fill="{BLUE}"/>')
        p.append(f'<rect x="{cx}" y="{y}" width="66" height="22" fill="{ORANGE}"/>')
    box(cx-66,y,132,22,'none',0)
    if spatial:line([(cx,y-3),(cx,y+25)],False,True)

for x,w,title in [(16,278,'(a) Residual'),(314,396,'(b) HC'),(730,494,'(c) SiHC')]:
    txt(x+w/2,25,title,21)
    line([(x,46),(x+w,46)],False)
    line([(x,490),(x+w,490)],False,True)

# Same vertical computational scaffold in all three panels.
for cx,n,spatial in [(155,1,False),(493,2,False),(923,2,True)]:
    states(cx,85,n,spatial,'ℓ+1')
    plus(cx,128);line([(cx,116),(cx,104)])
    states(cx,430,n,spatial)
    if n==1:line([(cx,448),(cx,450)],False)
    else:
        for sx in [cx-35,cx+35]:line([(sx,448),(sx,450)],False)
    txt(cx,480,'⋮',21)
    states(cx,526,n,spatial,'0',False)

# Plain residual.
line([(155,450),(56,450),(56,128),(143,128)])
line([(155,450),(238,450),(238,293)])
box(198,251,80,42,BLUE);txt(238,279,sub('f','ℓ'),23,math=True)
line([(238,251),(238,128),(167,128)])
line([(155,583),(155,546)])
box(100,583,110,32);txt(155,605,'Embedding',15)
line([(155,652),(155,616)])
patch(155,652);txt(155,698,'Patch',16)

# Static HC: scalar stream mappings and residual mixing.
left,right=385,607
line([(493,450),(left,450),(left,351)])
mat(left,315,'γ','2×2');txt(left+74,338,'Residual mixing',14,anchor='start')
line([(left,315),(left,128),(481,128)])
line([(493,450),(right,450),(right,397)])
mat(right,361,'α','1×2');txt(right-76,384,'Pooling',14,anchor='end')
line([(right,361),(right,325)])
feature(right,311,'z')
line([(right,311),(right,293)])
box(right-40,251,80,42,BLUE);txt(right,279,sub('f','ℓ'),23,math=True)
line([(right,251),(right,229)])
feature(right,215,'u')
line([(right,215),(right,199)])
mat(right,163,'β','2×1');txt(right-76,186,'Distribution',14,anchor='end')
line([(right,163),(right,128),(505,128)])
# The copied embedding is explicitly a single tensor before repeat.
line([(493,575),(493,546)])
txt(530,564,'Repeat',14,anchor='start')
box(460,575,66,12,GRAY)
line([(493,601),(493,588)])
box(438,601,110,31);txt(493,622,'Embedding',15)
line([(493,652),(493,633)])
patch(493,652);txt(493,698,'Patch',16)

# SiHC: the addition is a tensor addition, matching the HC scaffold.
left,right=786,1084
line([(923,450),(left,450),(left,128),(911,128)])
txt(803,291,'Identity',14,anchor='start')
line([(923,450),(right,450),(right,397)])
mat(right,361,'α','c×2');txt(right-76,384,'Channel-wise pooling',14,anchor='end')
line([(right,361),(right,325)])
feature(right,311,'z')
line([(right,311),(right,293)])
box(right-40,251,80,42,BLUE);txt(right,279,sub('f','ℓ'),23,math=True)
line([(right,251),(right,229)])
feature(right,215,'u')
line([(right,215),(right,199)])
mat(right,163,'β','c×2');txt(right-76,186,'Channel-wise distribution',14,anchor='end')
line([(right,163),(right,128),(935,128)])
# The two states correspond to the two subpatches, without crossing paths.
for cx,fill in [(888,BLUE),(958,ORANGE)]:
    line([(cx,589),(cx,546)])
    box(cx-31,589,62,31,fill);txt(cx,610,'Embed',13)
    line([(cx,652),(cx,621)])
patch(923,652,True)
txt(923,698,'Patch → two subpatches',16)
txt(620,735,'Each sublayer: attention or MLP   ·   c: feature dimension   ·   two streams shown',14)
p.append('</svg>')
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'sihc_comparison_v3.svg').write_text('\n'.join(p))
print(OUT/'sihc_comparison_v3.svg')

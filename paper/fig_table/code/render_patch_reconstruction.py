"""Reuse the mountain row of the archived PCA visualization at native resolution."""
from pathlib import Path
import numpy as np
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
ROOT=Path(__file__).resolve().parents[2]
source=ROOT/'fig_table/figure/section03_analysis/01_patch_embedding_matrix/patch_pca_visual_reconstruction.png'
out=ROOT/'fig_table/figure/manuscript_panels'
im=Image.open(source).convert('RGB')
a=np.asarray(im)
active=np.where(np.any(a<245,axis=2).sum(axis=1)>im.width*.3)[0]
groups=np.split(active,np.where(np.diff(active)>2)[0]+1)
groups=[g for g in groups if len(g)>100]
assert len(groups)==5
mountain=im.crop((0,int(groups[1][0]),im.width,int(groups[1][-1])+1))
width=408
height=mountain.height*width/mountain.width
c=canvas.Canvas(str(out/'patch_reconstruction_mountain.pdf'),pagesize=(width,height+13))
c.drawImage(ImageReader(mountain),0,13,width=width,height=height)
c.setFont('Helvetica',8)
for j,label in enumerate(['Original','Top 8','Top 16','Top 32','Top 64','Top 128']):
    c.drawCentredString((j+.5)*width/6,2,label)
c.save()
print(f'Mountain row embedded without resampling: {mountain.size}; other rows retained in the appendix source.')

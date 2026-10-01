"""Package the native architecture export as the manuscript PDF, without headers.

First run render_sihc_overview.mjs with the Artifact Tool Node.js runtime.
This wrapper places that export on a PDF page without changing its pixels.
"""
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

ROOT = Path(__file__).resolve().parents[3]
IMAGE = ROOT / "paper/figures/architecture/sihc_overview.png"
OUT = ROOT / "paper/figures/architecture/sihc_overview.pdf"
image = ImageReader(str(IMAGE))
width_px, height_px = image.getSize()
width = 540.0
height = width * height_px / width_px
c = canvas.Canvas(str(OUT), pagesize=(width, height))
c.setTitle("Residual connection, Hyper-Connections, and SiHC (ours)")
c.drawImage(image, 0, 0, width=width, height=height)
c.showPage()
c.save()
print("Packaged the architecture diagram without additional header labels.")

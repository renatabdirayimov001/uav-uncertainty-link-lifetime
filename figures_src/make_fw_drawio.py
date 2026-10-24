"""Figure 3 (RCLR framework) as an editable draw.io flowchart in a plain academic style:
white blocks, thin dark outlines, orthogonal connectors, Times New Roman, no icons."""
import make_drawio_lib as lib
from make_drawio_lib import D

F = 'Times New Roman'
FS = 17                                   # 860 px wide -> 5.46 in in print: 17 px ~ 8 pt
lib.FONT = f'fontFamily={F};fontSize={FS};'
INK = '#222222'
PLAIN = f'#ffffff;strokeColor={INK}'
CORE = f'#e8e8e8;strokeColor={INK}'       # components of the proposed bound
ACI = f'#ffffff;strokeColor={INK};dashed=1;dashPattern=6 3'
BOX = 'rounded=0;whiteSpace=wrap;html=1;strokeWidth=1.3;spacing=4;'
ARR = f'endArrow=block;endFill=1;endSize=7;strokeWidth=1.3;strokeColor={INK};fontFamily={F};fontSize={FS - 2};'
DASH = ARR + 'dashed=1;dashPattern=6 3;'

d = D('rclr-framework', 860, 560)
W, H = 186, 78
X = [20, 234, 448, 662]
WL = [W, W, W, W]

def lane(y, h, title):
    d.shape(f'rounded=0;html=1;fillColor=none;strokeColor={INK};strokeWidth=1;dashed=1;dashPattern=8 4;', 4, y, 852, h)
    d.text(title, 14, y + 4, 700, 28, size=FS, align='left', extra=f'fontFamily={F};fontStyle=1;')

# ---------------- offline phase
lane(4, 150, '(a) Offline phase: ground station, before the mission')
yo = 56
o1 = d.box('Flight logs or<br>simulated traces', X[0], yo, W, H, PLAIN, BOX + 'shape=cylinder3;boundedLbl=1;size=8;')
o2 = d.box('Link samples $(\\mathbf{x}_k,\\,y_k)$<br>(training and<br>calibration sets)', X[1], yo, W, H, PLAIN, BOX)
o3 = d.box('Lower-quantile model<br>$\\hat q_\\alpha$ (pinball loss)', X[2], yo, W, H, CORE, BOX)
o4 = d.box('Conformity scores<br>$E_k = \\hat q_\\alpha(\\mathbf{x}_k) - y_k$', X[3], yo, WL[3], H, CORE, BOX)
for a, b in [(o1, o2), (o2, o3), (o3, o4)]:
    d.edge(a, b, style=ARR)

# ---------------- online phase
lane(170, 382, '(b) Online phase: on board every UAV, every period $T_u$')
y1, y2 = 236, 440
n1 = d.box('Neighbour table<br>(HELLO, age $A_{ij}$)', X[0], y1, W, H, PLAIN, BOX)
n2 = d.box('Link features<br>$\\mathbf{x}_{ij}(t)$', X[1], y1, W, H, PLAIN, BOX)
n3 = d.box('Lower quantile<br>$\\hat q_\\alpha(\\mathbf{x}_{ij}(t))$', X[2], y1, W, H, CORE, BOX)
n4 = d.box('Calibrated bound<br>$L_{ij}$ = max{$\\hat q_\\alpha$ − $\\hat Q_{1-\\alpha_t}$, 0}', X[3], y1, WL[3], H, CORE, BOX + 'fontStyle=0;')
for a, b in [(n1, n2), (n2, n3), (n3, n4)]:
    d.edge(a, b, style=ARR)
# offline -> online transfers
d.edge(o3, n3, style=DASH); d.text('model $\\hat q_\\alpha$', X[2] + W / 2 + 6, 196, 90, 26, size=FS - 1, align='left', extra=f'fontFamily={F};')
d.edge(o4, n4, style=DASH); d.text('scores $\\{E_k\\}$', X[3] + W / 2 + 6, 196, 90, 26, size=FS - 1, align='left', extra=f'fontFamily={F};')

m3 = d.box('Route selection: widest<br>path; switch if $L(\\mathcal{P}) \\le T_u$', X[3], y2, W, H, PLAIN, BOX)
m2 = d.box('Observed link outcome<br>err$_t$ = 𝟙{$Y &lt; L$}', X[2], y2, W, H, PLAIN, BOX)
m1 = d.box('ACI update (optional)<br>$\\alpha_{t+1} = \\alpha_t + \\gamma\\,(\\alpha$ − err$_t$)', X[1] - 60, y2, W + 60, H, ACI, BOX)
d.edge(n4, m3, style=ARR + 'exitX=0.75;exitY=1;entryX=0.75;entryY=0;'); d.text('bounds $L_{ij}$', X[3] + 0.75 * W - 106, 360, 100, 26, size=FS - 1, align='right', extra=f'fontFamily={F};')
d.edge(m3, m2, style=ARR)
d.edge(m2, m1, style=ARR)
# feedback of the adapted level to the bound
d.edge(m1, n4, '$\\alpha_t$', style=DASH + 'exitX=0.5;exitY=0;exitDx=0;exitDy=0;entryX=0.25;entryY=1;entryDx=0;entryDy=0;labelBackgroundColor=#ffffff;',
       pts=[(X[1] - 60 + (W + 60) / 2, 375), (X[3] + W / 4, 375)])
# output to the network layer
d.text('route to the network layer', X[3], y2 + H + 4, W, 24, size=FS - 3, extra=f'fontFamily={F};fontStyle=2;')

d.save('fig_framework_academic.drawio')
x = open('fig_framework_academic.drawio').read().replace('fontFamily=Helvetica', f'fontFamily={F}')
open('fig_framework_academic.drawio', 'w').write(x)
print('ok')

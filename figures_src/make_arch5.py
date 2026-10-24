"""Isometric system architecture in the style of illustrated journal figures: two city tiles, a UAV swarm flying above,
a callout 'spotlight' from one UAV to its onboard module, GNSS and ground station, and a legend.
Artwork: Freepik (city tiles, drones, operator, tower, satellite); icons: Material Design Icons."""
import base64
import make_drawio_lib as lib
from make_drawio_lib import D
from make_arch2 import icon, image
lib.FONT = 'fontFamily=Helvetica;fontSize=17;'
def png(f): return 'data:image/png,' + base64.b64encode(open(f, 'rb').read()).decode()
BRICK, BLUEC, DR, OPER, TOWER, SAT = [png(f) for f in ['iso_city_brick.png', 'iso_city_blue.png', 'iso_drone.png', 'iso_operator.png', 'iso_tower.png', 'iso_sat.png']]
ENV = png('ic_envelope.png')
ICON = {k: png(f'ic_{k}.png') for k in ['hello', 'features', 'model', 'bound', 'route']}
def ic(d, uri, x, y, w, h):   # illustrated icon, aspect ratio kept
    return d.shape(f'shape=image;html=1;imageAspect=1;aspect=fixed;image={uri};', x, y, w, h)
DW, DH = 92, 57
def pic(d, uri, x, y, w, h, label='', size=17, extra=''):
    return d.shape(f'shape=image;html=1;imageAspect=0;image={uri};verticalLabelPosition=bottom;verticalAlign=top;labelPosition=center;align=center;'
                   f'fontFamily=Helvetica;fontSize={size};spacingTop=-4;{extra}', x, y, w, h, label)

d = D('system-architecture-iso', 1000, 910)
# ---- ground: two city tiles
d.shape('ellipse;fillColor=#000000;opacity=8;strokeColor=none;', 40, 700, 400, 70)
d.shape('ellipse;fillColor=#000000;opacity=8;strokeColor=none;', 530, 690, 420, 70)
pic(d, BRICK, 15, 470, 450, 311)
pic(d, BLUEC, 500, 440, 470, 330)
d.text('<b>Ground segment</b> (urban area)', 15, 792, 300, 24, size=17, align='left', extra='fontColor=#3d4a2a;')
# ---- swarm (centre coordinates), flying above the tiles
P = {'s': (95, 330, 'Source'), 'i': (245, 255, 'UAV $i$'), 'j': (385, 150, 'UAV $j$'), 'k': (480, 315, 'UAV $k$'),
     'm': (300, 400, ''), 'n': (640, 250, ''), 'd': (862, 350, 'Destination')}
c = lambda n: P[n][:2]
LONG = 'endArrow=none;strokeWidth=2;strokeColor=#7d8792;'
SHORT = 'endArrow=none;strokeWidth=2.5;strokeColor=#eb6834;dashed=1;dashPattern=6 4;'
ROUTE = 'endArrow=none;strokeWidth=6;strokeColor=#1baf7a;opacity=85;'
xi, yi = c('i'); xj, yj = c('j')
d.shape('ellipse;dashed=1;fillColor=#2a78d6;opacity=8;strokeColor=#2a78d6;strokeWidth=1.5;', xi - 175, yi - 85, 350, 170)
d.text('range $R$', xi + 62, yi + 58, 90, 22, size=17, align='left', extra='fontColor=#2a78d6;')
for a, b, st in [('s', 'm', LONG), ('m', 'k', LONG), ('i', 'm', LONG), ('i', 'j', SHORT), ('j', 'k', LONG), ('j', 'n', SHORT), ('n', 'd', LONG), ('k', 'n', LONG)]:
    d.line(*c(a), *c(b), style=st)
for a, b in [('s', 'i'), ('i', 'k'), ('k', 'd')]:
    d.line(*c(a), *c(b), style=ROUTE)
# departing j
d.line(xj + 22, yj - 10, xj + 80, yj - 52, style='endArrow=classic;strokeWidth=3;strokeColor=#eb6834;')
d.text('$\\mathbf{v}_j$', xj + 32, yj - 58, 34, 22, size=18, extra='fontColor=#eb6834;fontStyle=1;')
pic(d, DR, xj + 72, yj - 98, DW * 0.7, DH * 0.7, extra='opacity=40;')
d.text('link $(i,j)$ breaks<br>when $d_{ij}>R$', xj + 140, yj - 108, 170, 48, size=16, align='left')
hx, hy = (xi + xj) / 2 - 64, (yi + yj) / 2 - 36
ic(d, ENV, hx + 4, hy + 2, 40, 28)
d.text('HELLO, age $A_{ij}$', hx - 160, hy + 3, 158, 22, size=17, align='right')
# soft drop shadows on the ground make the swarm look airborne
for n, (x, y, lab) in P.items():
    if n in ('s', 'd'):
        d.shape('ellipse;fillColor=#1baf7a;opacity=30;strokeColor=#1baf7a;strokeWidth=2;', x - 42, y + 6, 84, 26)
    pic(d, DR, x - DW / 2, y - DH / 2 - 6, DW, DH, lab)
d.text('<b>UAV swarm (FANET)</b>', 480, 140, 220, 24, size=18, align='left', extra='fontColor=#24405e;')
# ---- GNSS
pic(d, SAT, 5, 5, 115, 68)
d.text('GNSS positioning (error $s_j$)', 118, 22, 260, 22, size=17, align='left', extra='fontColor=#55606e;')
d.line(55, 72, P['s'][0] - 5, P['s'][1] - 30, style='endArrow=none;dashed=1;strokeColor=#9aa3ad;strokeWidth=1;')
# ---- ground station between the tiles
pic(d, TOWER, 448, 646, 44, 100)
pic(d, OPER, 490, 694, 100, 92)
d.text('<b>Ground station</b>: offline training<br>and calibration of $\\hat q_\\alpha$', 340, 786, 340, 48, size=16)
d.line(470, 646, P['m'][0] + 15, P['m'][1] + 30, style='endArrow=classic;dashed=1;strokeWidth=2;strokeColor=#2b3a4a;')
# ---- spotlight callout from UAV n to its onboard module (like the 'block' callouts of illustrated figures)
xn, yn = c('d'); BX, BY, BW, BH = 688, 8, 308, 226
d.shape('triangle;direction=south;html=1;fillColor=#9aa3ad;opacity=28;strokeColor=none;', BX + 45, BY + BH, BW - 90, yn - 30 - (BY + BH))
d.shape('rounded=1;arcSize=8;html=1;dashed=1;fillColor=#fbecf3;strokeColor=#d98cb3;strokeWidth=1.5;', BX, BY, BW, BH)
d.text('<b>Onboard RCLR module</b>', BX, BY + 4, BW, 24, size=17, extra='fontColor=#1f5fae;')
rows = [('hello', 'HELLO table, age $A_{ij}$'), ('features', 'features $\\mathbf{x}_{ij}(t)$'), ('model', 'lower quantile $\\hat q_\\alpha$'),
        ('bound', 'bound $L_{ij}$, target coverage $1-\\alpha$'), ('route', 'widest path, switch if $L\\le T_u$')]
for k, (icn, txt) in enumerate(rows):
    y = BY + 34 + k * 37
    ic(d, ICON[icn], BX + 8, y - 1, 44, 34)
    d.text(txt, BX + 56, y, BW - 60, 32, size=16, align='left')
# ---- legend
LY = 846
d.shape('rounded=1;arcSize=10;html=1;dashed=1;fillColor=#fbecf3;strokeColor=#d98cb3;strokeWidth=1.5;', 15, LY, 970, 56)
pic(d, DR, 22, LY + 10, 56, 35); d.text('UAV', 80, LY + 17, 50, 22, size=17, align='left')
d.shape('ellipse;fillColor=#1baf7a;opacity=30;strokeColor=#1baf7a;strokeWidth=2;', 131, LY + 32, 46, 15); pic(d, DR, 126, LY + 10, 56, 35)
d.text('source / destination', 184, LY + 17, 170, 22, size=17, align='left')
d.line(358, LY + 29, 393, LY + 29, style=ROUTE); d.text('selected route', 398, LY + 17, 125, 22, size=17, align='left')
d.line(530, LY + 29, 565, LY + 29, style=LONG); d.text('long bound', 570, LY + 17, 100, 22, size=17, align='left')
d.line(675, LY + 29, 710, LY + 29, style=SHORT); d.text('link at risk', 715, LY + 17, 100, 22, size=17, align='left')
ic(d, ENV, 818, LY + 15, 40, 28); d.text('HELLO', 862, LY + 17, 80, 22, size=17, align='left')
d.save('fig_system_model_iso3.drawio')
x = open('fig_system_model_iso3.drawio').read().replace('fontFamily=Helvetica', 'fontFamily=Times New Roman')
open('fig_system_model_iso3.drawio', 'w').write(x); print('ok')

"""System architecture, drawn at print scale (1000 px wide, used at full page width -> 16 px = 8 pt)."""
import base64, os
import make_drawio_lib as lib
from make_drawio_lib import D
from drone_svg import drone_svg
lib.FONT = 'fontFamily=Helvetica;fontSize=16;'
MDI = os.environ.get('MDI_DIR', 'node_modules/@mdi/svg/svg/')
def icon(name, color):
    s = open(MDI + name + '.svg').read().replace('<svg ', f'<svg fill="{color}" ')
    return 'data:image/svg+xml,' + base64.b64encode(s.encode()).decode()
def img(svg): return 'data:image/svg+xml,' + base64.b64encode(svg.encode()).decode()
def image(d, uri, x, y, w, h, label='', size=16, extra=''):
    return d.shape(f'shape=image;html=1;imageAspect=0;aspect=fixed;image={uri};verticalLabelPosition=bottom;verticalAlign=top;'
                   f'labelPosition=center;align=center;fontFamily=Helvetica;fontSize={size};spacingTop=-6;{extra}', x, y, w, h, label)
BLUE, GREEN, ORANGE = '#2a78d6', '#1baf7a', '#eb6834'
DRONE = {c: img(drone_svg(accent=c)) for c in [BLUE, GREEN, ORANGE]}
DW, DH = 84, 55

def build(path):
    d = D('system-architecture', 1000, 640)
    # planes
    d.shape('shape=parallelogram;perimeter=parallelogramPerimeter;html=1;fixedSize=1;size=90;fillColor=#e8f1fb;strokeColor=#8fb6e3;strokeWidth=1.5;'
            'gradientColor=#f7fbff;gradientDirection=north;', 10, 78, 640, 352)
    d.shape('shape=parallelogram;perimeter=parallelogramPerimeter;html=1;fixedSize=1;size=60;fillColor=#edf3e4;strokeColor=#b2c993;strokeWidth=1.5;'
            'gradientColor=#f8fbf3;gradientDirection=north;', 10, 488, 640, 140)
    d.text('<b>UAV swarm (FANET)</b>', 110, 84, 260, 24, size=17, align='left', extra='fontColor=#24405e;')
    d.text('<b>Ground segment</b>', 80, 494, 200, 24, size=17, align='left', extra='fontColor=#3d5a2a;')
    # GNSS
    image(d, icon('satellite-variant', '#55606e'), 520, 6, 46, 46)
    d.text('GNSS (error $s_j$)', 568, 16, 150, 24, size=16, align='left', extra='fontColor=#55606e;')
    P = {'s': (120, 345, GREEN, 'Source'), 'i': (245, 270, BLUE, 'UAV $i$'), 'j': (370, 185, BLUE, 'UAV $j$'),
         'k': (440, 300, BLUE, 'UAV $k$'), 'd': (575, 220, GREEN, 'Destination'), 'a': (300, 375, BLUE, '')}
    c = lambda n: P[n][:2]
    xi, yi = c('i'); xj, yj = c('j')
    d.shape('ellipse;dashed=1;fillColor=#2a78d6;opacity=7;strokeColor=#2a78d6;strokeWidth=1.5;', xi - 175, yi - 95, 350, 190)
    d.text('range $R$', xi - 165, yi + 12, 90, 22, size=16, align='left', extra='fontColor=#2a78d6;')
    LONG = 'endArrow=none;strokeWidth=2;strokeColor=#9aa3ad;'
    SHORT = 'endArrow=none;strokeWidth=2.5;strokeColor=#eb6834;dashed=1;dashPattern=6 4;'
    ROUTE = 'endArrow=none;strokeWidth=6;strokeColor=#1baf7a;opacity=85;'
    for a, b, st in [('s', 'a', LONG), ('a', 'k', LONG), ('i', 'a', LONG), ('i', 'j', SHORT), ('j', 'k', LONG), ('j', 'd', SHORT)]:
        d.line(*c(a), *c(b), style=st)
    for a, b in [('s', 'i'), ('i', 'k'), ('k', 'd')]:
        d.line(*c(a), *c(b), style=ROUTE)
    # departing j
    d.line(xj + 22, yj - 10, xj + 78, yj - 52, style='endArrow=classic;strokeWidth=3;strokeColor=#eb6834;')
    d.text('$\\mathbf{v}_j$', xj + 34, yj - 58, 34, 22, size=17, extra='fontColor=#eb6834;fontStyle=1;')
    image(d, DRONE[ORANGE], xj + 70, yj - 100, DW * 0.75, DH * 0.75, extra='opacity=35;')
    d.text('breaks when $d_{ij}>R$', xj + 128, yj - 100, 170, 22, size=16, align='left')
    # GPS blur under j, HELLO
    hx, hy = (xi + xj) / 2 - 62, (yi + yj) / 2 - 40
    image(d, icon('email-fast', '#c79a1e'), hx, hy, 34, 34)
    d.text('HELLO, age $A_{ij}$', hx - 150, hy + 4, 148, 24, size=16, align='right')
    for n, (x, y, col, lab) in P.items():
        image(d, DRONE[col], x - DW / 2, y - DH / 2, DW, DH, lab)
    # legend
    d.line(210, 455, 245, 455, style=ROUTE); d.text('selected route', 250, 444, 120, 22, size=16, align='left')
    d.line(375, 455, 410, 455, style=LONG); d.text('long bound', 415, 444, 100, 22, size=16, align='left')
    d.line(520, 455, 555, 455, style=SHORT); d.text('link at risk', 560, 444, 100, 22, size=16, align='left')
    # ground
    image(d, icon('radio-tower', '#33475b'), 100, 528, 50, 50)
    image(d, icon('laptop', '#33475b'), 152, 548, 34, 34)
    d.text('<b>Ground station:</b> offline training<br>and calibration of $\\hat q_\\alpha$', 192, 530, 270, 50, size=16, align='left')
    for nm, x, y, s in [('office-building', 440, 520, 44), ('office-building', 482, 540, 36), ('home-city', 525, 560, 40), ('pine-tree', 410, 572, 32)]:
        image(d, icon(nm, '#8fa38a'), x, y, s, s)
    d.line(140, 526, 125, 392, style='endArrow=classic;dashed=1;strokeWidth=2;strokeColor=#33475b;')
    d.text('model $\\hat q_\\alpha$, scores $\\{E_k\\}$', 138, 470, 230, 22, size=15, align='left', extra='fontColor=#33475b;')
    # onboard module
    X, Y, W = 680, 60, 310
    d.shape('rounded=1;arcSize=4;html=1;fillColor=#ffffff;strokeColor=#2a78d6;strokeWidth=2;', X, Y, W, 570)
    d.text('<b>Onboard RCLR module</b>', X, Y + 8, W, 26, size=17, extra='fontColor=#1f5fae;')
    steps = [('access-point', 'Neighbour table<br>(HELLO, age $A_{ij}$)'),
             ('table', 'Link features $\\mathbf{x}_{ij}(t)$'),
             ('chip', 'Lower quantile $\\hat q_\\alpha(\\mathbf{x}_{ij})$'),
             ('shield-check', 'Calibrated bound $L_{ij}$<br>$\\Pr\\{Y_{ij}\\ge L_{ij}\\}\\ge1-\\alpha$'),
             ('routes', 'Widest path, switch<br>if $L(\\mathcal{P})\\le T_u$'),
             ('sync', 'Update $\\alpha_t$ (optional)')]
    for k, (ic, txt) in enumerate(steps):
        y = Y + 44 + k * 87
        d.shape('rounded=1;html=1;fillColor=#f2f7fd;strokeColor=#b7cfee;', X + 12, y, W - 24, 66)
        image(d, icon(ic, BLUE), X + 22, y + 15, 36, 36)
        d.text(txt, X + 66, y + 3, W - 84, 60, size=16, align='left')
        if k: d.line(X + W / 2, y - 21, X + W / 2, y, style='endArrow=block;endFill=1;strokeWidth=2;strokeColor=#2a78d6;')
    d.shape('ellipse;fillColor=none;strokeColor=#2a78d6;strokeWidth=2;dashed=1;', xi - 50, yi - 34, 100, 68)
    d.line(xi + 50, yi - 8, X, Y + 30, style='endArrow=none;dashed=1;strokeColor=#2a78d6;strokeWidth=1.5;')
    d.save(path)

if __name__ == '__main__':
    build('fig_system_model.drawio'); print('ok')

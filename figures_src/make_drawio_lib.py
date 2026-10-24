"""Generates editable draw.io diagrams for the paper (open in app.diagrams.net -> File -> Open)."""
from xml.sax.saxutils import escape
import os


import re as _re
_MAP = [(r'\mathcal{P}', '𝒫'), (r'\mathcal{A}', '𝒜'), (r'\mathcal{I}', 'ℐ'), (r'\mathrm{err}', 'err'),
        (r'\alpha', 'α'), (r'\gamma', 'γ'), (r'\kappa', 'κ'), (r'\pi', 'π'), (r'\ge', '≥'), (r'\le', '≤'),
        (r'\Pr', 'Pr'), (r'\{', '\x01'), (r'\}', '\x02'), (r'\min', 'min '), (r'\max', 'max'), ('\\,', ' ')]
def tex2html(t):
    for a, b in _MAP: t = t.replace(a, b)
    t = _re.sub(r'\\mathbf\{(\w+)\}', r'<b>\1</b>', t)
    t = _re.sub(r'\\hat\s*(\w)', lambda m: m.group(1) + '\u0302', t)
    t = t.replace('-', '−')
    for _ in range(3):
        t = _re.sub(r'_\{([^{}]*)\}', r'<sub>\1</sub>', t)
        t = _re.sub(r'_(\w|α|𝒫)', r'<sub>\1</sub>', t)
        t = _re.sub(r'\^\{([^{}]*)\}', r'<sup>\1</sup>', t)
    t = t.replace('{', '').replace('}', '') if '\\' not in t else t
    t = t.replace('\x01', '{').replace('\x02', '}')
    return '<i>' + t + '</i>'

BLUE = '#dae8fc;strokeColor=#2a78d6'
GREY = '#f5f5f5;strokeColor=#666666'
ORNG = '#fde7dc;strokeColor=#eb6834'
GRN = '#d5f0e6;strokeColor=#1baf7a'
FONT = 'fontFamily=Helvetica;fontSize=13;'


class D:
    def __init__(self, name, w=1100, h=700):
        self.name, self.w, self.h, self.cells, self.n = name, w, h, [], 2

    def _id(self):
        self.n += 1
        return f'c{self.n}'

    def box(self, label, x, y, w, h, fill=GREY, extra='rounded=1;whiteSpace=wrap;html=1;'):
        i = self._id()
        self.cells.append(f'<mxCell id="{i}" value="{escape(_re.sub(r'\$([^$]+)\$', lambda m: tex2html(m.group(1)), label), {chr(34): "&quot;"})}" style="{extra}fillColor={fill};{FONT}" vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        return i

    def text(self, label, x, y, w, h, size=13, extra='', align='center'):
        i = self._id()
        self.cells.append(f'<mxCell id="{i}" value="{escape(_re.sub(r'\$([^$]+)\$', lambda m: tex2html(m.group(1)), label), {chr(34): "&quot;"})}" style="text;html=1;align={align};verticalAlign=middle;whiteSpace=wrap;fontFamily=Helvetica;fontSize={size};{extra}" vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        return i

    def shape(self, style, x, y, w, h, label=''):
        i = self._id()
        self.cells.append(f'<mxCell id="{i}" value="{escape(_re.sub(r'\$([^$]+)\$', lambda m: tex2html(m.group(1)), label), {chr(34): "&quot;"})}" style="{style}" vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        return i

    def edge(self, s, t, label='', style='endArrow=block;endFill=1;strokeWidth=2;strokeColor=#444444;', pts=None):
        i = self._id()
        p = ''
        if pts:
            p = '<Array as="points">' + ''.join(f'<mxPoint x="{a}" y="{b}"/>' for a, b in pts) + '</Array>'
        self.cells.append(f'<mxCell id="{i}" value="{escape(_re.sub(r'\$([^$]+)\$', lambda m: tex2html(m.group(1)), label), {chr(34): '&quot;'})}" style="{style}edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;fontSize=16;" edge="1" parent="1" source="{s}" target="{t}"><mxGeometry relative="1" as="geometry">{p}</mxGeometry></mxCell>')
        return i

    def line(self, x1, y1, x2, y2, style='endArrow=block;endFill=1;strokeWidth=2;strokeColor=#444444;', label=''):
        i = self._id()
        self.cells.append(f'<mxCell id="{i}" value="{escape(_re.sub(r'\$([^$]+)\$', lambda m: tex2html(m.group(1)), label), {chr(34): '&quot;'})}" style="{style}html=1;fontSize=16;" edge="1" parent="1"><mxGeometry relative="1" as="geometry"><mxPoint x="{x1}" y="{y1}" as="sourcePoint"/><mxPoint x="{x2}" y="{y2}" as="targetPoint"/></mxGeometry></mxCell>')
        return i

    def save(self, path):
        import re
        body = ''.join(self.cells)
        xml = (f'<mxfile host="app.diagrams.net"><diagram name="{self.name}" id="{self.name}">'
               f'<mxGraphModel dx="1200" dy="800" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{self.w}" pageHeight="{self.h}" math="0" shadow="0">'
               f'<root><mxCell id="0"/><mxCell id="1" parent="0"/>{body}</root></mxGraphModel></diagram></mxfile>')
        open(path, 'w').write(xml)


def uav(d, x, y, label, fill='#2a78d6'):
    """simple quadrotor glyph: body + 4 rotors; returns id of body"""
    for dx, dy in [(-16, -16), (16, -16), (-16, 16), (16, 16)]:
        d.shape(f'ellipse;fillColor=none;strokeColor={fill};strokeWidth=1.5;', x + dx - 9, y + dy - 9, 18, 18)
    b = d.shape(f'rounded=1;fillColor={fill};strokeColor={fill};', x - 9, y - 9, 18, 18)
    d.text(label, x - 40, y + 26, 80, 20, size=13)
    return b



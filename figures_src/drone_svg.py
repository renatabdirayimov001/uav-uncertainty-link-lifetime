"""Vector quadcopter illustration (3/4 view) used as an image in the draw.io figures."""

def drone_svg(accent='#2a78d6', body='#3d4450', w=200, h=130):
    rot = [(-64, 22, 1.0), (64, 22, 1.0), (-46, -14, 0.82), (46, -14, 0.82)]   # (x, y, scale) front then back
    def rotor(x, y, s, front):
        rx, ry = 34 * s, 8.5 * s
        return (f'<line x1="0" y1="2" x2="{x}" y2="{y + 4}" stroke="{body}" stroke-width="{7 * s:.1f}" stroke-linecap="round"/>'
                f'<rect x="{x - 5 * s:.1f}" y="{y - 2:.1f}" width="{10 * s:.1f}" height="{12 * s:.1f}" rx="2" fill="url(#mot)"/>'
                f'<ellipse cx="{x}" cy="{y - 3}" rx="{rx:.1f}" ry="{ry:.1f}" fill="url(#disc)" stroke="#9aa3ad" stroke-width="0.8"/>'
                f'<line x1="{x - rx * 0.92:.1f}" y1="{y - 3 + ry * 0.25:.1f}" x2="{x + rx * 0.92:.1f}" y2="{y - 3 - ry * 0.25:.1f}" stroke="#58606b" stroke-width="{2.2 * s:.1f}" stroke-linecap="round" opacity="0.85"/>'
                f'<circle cx="{x}" cy="{y - 3}" r="{2.6 * s:.1f}" fill="#22262c"/>')
    back = ''.join(rotor(x, y, s, False) for x, y, s in rot[2:])
    front = ''.join(rotor(x, y, s, True) for x, y, s in rot[:2])
    core = (f'<path d="M-9 14 L-16 36 M9 14 L16 36 M-22 36 L-10 36 M10 36 L22 36" stroke="#5b6370" stroke-width="3" stroke-linecap="round" fill="none"/>'
            f'<ellipse cx="0" cy="4" rx="27" ry="13" fill="url(#bod)"/>'
            f'<ellipse cx="0" cy="-1" rx="19" ry="7.5" fill="url(#top)"/>'
            f'<rect x="-7" y="13" width="14" height="10" rx="4" fill="#22262c"/><circle cx="0" cy="18" r="3.2" fill="#6fa8ff"/>'
            f'<ellipse cx="0" cy="8" rx="20" ry="2.2" fill="{accent}"/>')
    defs = (f'<defs><radialGradient id="disc" cx="50%" cy="40%" r="60%"><stop offset="0" stop-color="#ffffff" stop-opacity="0.75"/>'
            f'<stop offset="1" stop-color="#c9d1da" stop-opacity="0.45"/></radialGradient>'
            f'<linearGradient id="bod" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#6b7480"/><stop offset="1" stop-color="{body}"/></linearGradient>'
            f'<linearGradient id="top" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#f4f6f8"/><stop offset="1" stop-color="#b9c0c8"/></linearGradient>'
            f'<linearGradient id="mot" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#2a2f36"/><stop offset="0.5" stop-color="#7a828c"/><stop offset="1" stop-color="#2a2f36"/></linearGradient></defs>')
    shadow = ''
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="-100 -40 200 82">{defs}'
            f'<g>{back}{core}{front}</g></svg>')

def to_drawio_image(svg):
    import base64
    return 'data:image/svg+xml,' + base64.b64encode(svg.encode()).decode()

if __name__ == '__main__':
    import sys
    open(sys.argv[1] if len(sys.argv) > 1 else 'drone.svg', 'w').write(drone_svg())

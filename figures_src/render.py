"""Render .drawio files to PDF/PNG with the draw.io static viewer in headless Chromium (local, offline)."""
import sys, json, html, asyncio, pathlib
from playwright.async_api import async_playwright

SCR = '/tmp/claude-0/-home-claude/3e998bd2-6b24-564b-96a8-e7d06814923b/scratchpad'
VIEWER = f'{SCR}/dio/src/main/webapp/js/viewer-static.min.js'
MATH = f'{SCR}/mj/node_modules/mathjax'

async def render(src, out_pdf, out_png):
    xml = pathlib.Path(src).read_text()
    import re as _re
    xml = _re.search(r'<mxGraphModel.*</mxGraphModel>', xml, _re.S).group(0)
    cfg = json.dumps({'xml': xml, 'math': True, 'border': 10})
    page_html = f"""<html><head><meta charset="utf-8">
<script>window.DRAW_MATH_URL='http://127.0.0.1:8765/mj/node_modules/mathjax';</script>
<style>body{{margin:0;background:#fff}}</style></head><body>
<div class="mxgraph" data-mxgraph="{html.escape(cfg)}"></div>
<script src="http://127.0.0.1:8765/dio/src/main/webapp/js/viewer-static.min.js"></script></body></html>"""
    tmp = pathlib.Path(SCR) / (pathlib.Path(src).stem + '.html'); tmp.write_text(page_html)
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path='/opt/pw-browsers/chromium-1194/chrome-linux/chrome', args=['--allow-file-access-from-files'])
        pg = await b.new_page(viewport={'width': 1400, 'height': 1000}, device_scale_factor=2)
        await pg.goto('http://127.0.0.1:8765/' + tmp.name)
        await pg.wait_for_timeout(12000)
        box = await pg.evaluate("(()=>{const s=document.querySelector('.mxgraph svg');const r=s.getBoundingClientRect();return [r.width,r.height]})()")
        w, h = int(box[0]) + 4, int(box[1]) + 4
        await pg.set_viewport_size({'width': w, 'height': h})
        await pg.screenshot(path=out_png, clip={'x': 0, 'y': 0, 'width': w, 'height': h})
        await pg.pdf(path=out_pdf, width=f'{w}px', height=f'{h}px', print_background=True, page_ranges='1')
        await b.close()

if __name__ == '__main__':
    for f in sys.argv[1:]:
        stem = pathlib.Path(f).with_suffix('')
        asyncio.run(render(f, str(stem) + '.pdf', str(stem) + '.png'))
        print('rendered', stem)

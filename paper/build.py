"""build.py — 提出物を paper/dist/ に作る。

    ~/kaggriculture/.venv/bin/python paper/build.py

paper.pdf            長い版（Project Link 用）。paper.md を HTML にして Chrome で印刷する
writeup_body.md      Kaggle の Writeup に貼る本文（題・副題・コメントを除く。図はアップロードして差し替える）
"""
import base64, os, re, subprocess, tempfile
from markdown_it import MarkdownIt

HERE = os.path.dirname(os.path.abspath(__file__))
DIST = os.path.join(HERE, 'dist')
CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
CSS = """
@page { size: A4; margin: 20mm 18mm; }
body { font-family: Georgia, 'Times New Roman', serif; font-size: 10.5pt; line-height: 1.42; color: #111; }
h1 { font-size: 19pt; line-height: 1.2; margin: 0 0 10pt; }
h2 { font-size: 13pt; margin: 16pt 0 5pt; } h3 { font-size: 11pt; margin: 12pt 0 4pt; }
h2, h3 { break-after: avoid; font-family: Helvetica, Arial, sans-serif; }
table { border-collapse: collapse; margin: 8pt 0; font-size: 9pt; font-family: Helvetica, Arial, sans-serif; break-inside: avoid; }
th, td { border-bottom: 0.5pt solid #999; padding: 3pt 7pt; text-align: left; vertical-align: top; }
th { border-bottom: 1pt solid #333; }
img { max-width: 100%; display: block; margin: 8pt auto; }
code { font-family: Menlo, monospace; font-size: 0.86em; }
a { color: #123f7a; word-break: break-all; } hr { display: none; }
"""


def inline_images(html):
    def rep(m):
        data = open(os.path.join(HERE, m.group(1)), 'rb').read()
        return 'src="data:image/png;base64,' + base64.b64encode(data).decode() + '"'
    return re.sub(r'src="(fig/[^"]+)"', rep, html)


def main():
    os.makedirs(DIST, exist_ok=True)
    s = open(os.path.join(HERE, 'paper.md'), encoding='utf-8').read()
    todo = s.count('[TODO')
    s = re.sub(r'<!--.*?-->', '', s, flags=re.S)
    s = re.sub(r'^\*(Draft|Version) .*\n', '', s, flags=re.M)
    html = MarkdownIt('commonmark', {'html': True}).enable('table').render(s)
    html = inline_images(html)
    title = re.match(r'# (.+)', s).group(1)
    page = f'<!doctype html><html lang="en"><meta charset="utf-8"><title>{title}</title><style>{CSS}</style><body>{html}</body></html>'
    with tempfile.NamedTemporaryFile('w', suffix='.html', delete=False, encoding='utf-8') as fh:
        fh.write(page)
    out = os.path.join(DIST, 'paper.pdf')
    subprocess.run([CHROME, '--headless=new', '--disable-gpu', '--no-pdf-header-footer',
                    f'--print-to-pdf={out}', 'file://' + fh.name], check=True, capture_output=True)
    os.remove(fh.name)
    w = open(os.path.join(HERE, 'writeup.md'), encoding='utf-8').read()
    w = re.sub(r'<!--.*?-->\n*', '', w, flags=re.S)
    w = re.sub(r'^# .*\n+### .*\n+', '', w)
    open(os.path.join(DIST, 'writeup_body.md'), 'w', encoding='utf-8').write(w)
    print(out, os.path.getsize(out), 'bytes;', f'TODO {todo} 件')


if __name__ == '__main__':
    main()

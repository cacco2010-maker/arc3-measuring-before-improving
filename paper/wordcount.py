"""wordcount.py — Kaggle Writeup の語数（上限 1,500）を数える。HTML コメントと画像は数えない。

    python3 paper/wordcount.py [paper/writeup.md]
"""
import re, sys

f = sys.argv[1] if len(sys.argv) > 1 else __file__.rsplit('/', 1)[0] + '/writeup.md'
s = open(f, encoding='utf-8').read()
s = re.sub(r'<!--.*?-->', '', s, flags=re.S)
s = re.sub(r'!\[[^\]]*\]\([^)]*\)', '', s)
total = 0
for sec in re.split(r'\n(?=## )', s):
    n = len(re.findall(r"[A-Za-z0-9][\w.'’′±×%/·-]*", sec))
    total += n
    print(f'{n:5d}  {sec.strip().splitlines()[0][:60]}')
print(f'{total:5d}  total (limit 1,500)')

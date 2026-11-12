"""Build the upload package in submission/: one self-contained .tex per document, as the Springer Nature template asks.

Inlines \\tabinput tables and \\input TikZ figures, writes pgfplots .dat files as inline tables, resolves \\sref
cross-document references from supplementary.aux, and strips every LaTeX comment (author notes and
[HUMAN REVIEW REQUIRED] markers must not reach the editor). Run after compiling supplementary.tex:
    python code/flatten_submission.py
"""
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'submission'


def strip_comments(text):
    lines = []
    for line in text.splitlines():
        if re.match(r'\s*%', line):
            continue  # a whole-line comment adds nothing to the output
        lines.append(re.sub(r'(?<!\\)%.*$', '%', line))  # keep the %, which also suppresses the line-end space
    return '\n'.join(lines) + '\n'


def inline_dat(match):
    opts, path = match.group(1), match.group(2)
    rows = (ROOT / path).read_text().strip().splitlines()
    return f'table[{opts}, row sep=\\\\]{{' + ' \\\\ '.join(rows) + ' \\\\}'


def flatten(text, labels):
    text = strip_comments(text)
    text = re.sub(r'\\tabinput\{([^}]+)\}', lambda m: strip_comments((ROOT / m.group(1)).read_text()).strip(), text)
    text = re.sub(r'\\input\{(figures/tikz/[^}]+)\}', lambda m: strip_comments((ROOT / m.group(1)).read_text()).strip(), text)
    text = re.sub(r'table\[([^\]]*)\]\{(results/revision/[^}]+\.dat)\}', inline_dat, text)
    text = re.sub(r'\\sref\{([^}]+)\}', lambda m: labels[m.group(1)], text)
    for macro in [r'\usepackage{xr}', r'\externaldocument[S-]{supplementary}',
                  r'\makeatletter\newcommand{\tabinput}[1]{\@@input #1 }\makeatother', r'\newcommand{\sref}[1]{\ref{S-#1}}']:
        text = text.replace(macro + '\n', '')
    assert '\\tabinput' not in text and '\\sref' not in text, 'unresolved input'
    assert not re.search(r'\{(results/revision|figures/tikz)/[^}]*\.(dat|tex)\}', text), 'unresolved file input'
    return strip_comments(text)


def main():
    OUT.mkdir(exist_ok=True)
    aux = (ROOT / 'supplementary.aux').read_text()
    labels = dict(re.findall(r'\\newlabel\{([^}]+)\}\{\{([^}]*)\}', aux))
    for name in ['main.tex', 'supplementary.tex', 'title_page.tex']:
        (OUT / name).write_text(flatten((ROOT / name).read_text(encoding='utf-8'), labels), encoding='utf-8')
    for name in ['references.bib', 'sn-jnl.cls', 'sn-basic.bst']:
        shutil.copy(ROOT / name, OUT / name)
    (OUT / 'figures').mkdir(exist_ok=True)
    shutil.copy(ROOT / 'figures' / 'disability_journey.pdf', OUT / 'figures' / 'disability_journey.pdf')
    print('wrote', sorted(p.name for p in OUT.iterdir()))


if __name__ == '__main__':
    main()

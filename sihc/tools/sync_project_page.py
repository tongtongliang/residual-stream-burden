"""Export a built blog to docs/ for GitHub Pages; never publish writing drafts.

From the release repository:
  python sihc/tools/sync_project_page.py --source ../elucidating_residual_stream_burden/blog
Then review, commit and push docs/. GitHub Pages deploys main:/docs automatically.
"""
import argparse
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.refs = []
    def handle_starttag(self, tag, attrs):
        self.refs.extend(v for k, v in attrs if k in ('href', 'src') and v)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    destination = Path(__file__).resolve().parents[2] / 'docs'
    index = (source / 'index.html').read_text()
    index = index.replace('../paper/preprint/output/pdf/main.pdf', 'https://arxiv.org/abs/2609.33895')
    pending = ['index.html', 'assets/katex/LICENSE']
    files = {}
    while pending:
        name = pending.pop()
        if name in files:
            continue
        path = (source / name).resolve()
        if not path.is_relative_to(source):
            raise ValueError(f'Reference escapes blog directory: {name}')
        data = index.encode() if name == 'index.html' else path.read_bytes()
        files[name] = data
        refs = []
        if path.suffix in ('.html', '.css', '.svg'):
            text = data.decode()
            links = Links()
            links.feed(text)
            refs.extend(links.refs)
            refs.extend(re.findall(r'url\([\'\"]?([^\)\'\"]+)', text))
        for ref in refs:
            parsed = urlsplit(ref)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            target = (path.parent / unquote(parsed.path)).resolve()
            if not target.is_relative_to(source):
                raise ValueError(f'Local reference outside blog: {ref}')
            pending.append(str(target.relative_to(source)))
    files['.nojekyll'] = b''
    manifest = destination / '.project-page-files.json'
    old = json.loads(manifest.read_text()) if manifest.exists() else []
    for name in old:
        stale = (destination / name).resolve()
        if not stale.is_relative_to(destination.resolve()):
            raise ValueError('Invalid publication manifest path')
        if name not in files and stale.is_file():
            stale.unlink()
    for name, data in files.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    manifest.write_text(json.dumps(sorted(files), indent=2) + '\n')
    print(f'Published {len(files)} static files to {destination}; source blog unchanged.')


if __name__ == '__main__':
    main()

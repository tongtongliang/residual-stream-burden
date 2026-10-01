#!/usr/bin/env python3
"""Build an audited, source-only anonymous ZIP with normalized metadata."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import stat
import tempfile
import zipfile

from audit_release import audit_tree

DIRECTORIES = ('sihc', 'training', 'evaluation', 'research', 'data', 'bench',
               'configs', 'docs', 'tests', 'tools', 'LICENSES')
ROOT_FILES = ('README.md', 'SUPPLEMENT.md', 'LICENSE', 'THIRD_PARTY_NOTICES.md',
              'pyproject.toml', 'requirements-tested.txt', '.gitignore')
SUFFIXES = {'.py', '.md', '.json', '.toml', '.txt', '.csv'}
SKIP = {'__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache', '.cache'}


def build(root: Path, output: Path, denylist: tuple[str, ...] = ()) -> dict:
    root, output = root.resolve(), output.resolve()
    if root == output or root in output.parents:
        raise ValueError('Write the archive outside the source repository.')
    selected = []
    for name in ROOT_FILES:
        path = root/name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f'Missing or nonregular release file: {name}')
        selected.append(path)
    for directory in DIRECTORIES:
        base = root/directory
        if base.is_symlink() or not base.is_dir():
            raise ValueError(f'Missing or linked release directory: {directory}')
        for path in sorted(base.rglob('*')):
            relative = path.relative_to(base)
            if any(part in SKIP for part in relative.parts):
                continue
            if path.is_symlink():
                raise ValueError(f'Symlink is not release source: {path.relative_to(root)}')
            if path.is_dir():
                continue
            if path.suffix not in SUFFIXES or not path.is_file():
                raise ValueError(f'Unreviewed artifact in release source: {path.relative_to(root)}')
            selected.append(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='anonymous-source-') as temporary:
        stage = Path(temporary)/'sihc-supplement'
        stage.mkdir()
        inventory = []
        for path in selected:
            relative = path.relative_to(root)
            target = stage/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            inventory.append(dict(path=relative.as_posix(), bytes=target.stat().st_size))
        (stage/'SOURCE_INVENTORY.json').write_text(json.dumps(inventory, indent=2)+'\n')
        findings = audit_tree(stage, denylist=denylist)
        if findings:
            # The scanner reports locations and rule names, never matching values.
            locations = '\n'.join(f'{f.path}:{f.line}: {f.rule}' for f in findings)
            raise ValueError('Anonymous source audit failed:\n'+locations)
        pending = output.with_suffix(output.suffix+'.tmp')
        try:
            with zipfile.ZipFile(pending, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
                for path in sorted(stage.rglob('*')):
                    if not path.is_file():
                        continue
                    entry = zipfile.ZipInfo(path.relative_to(stage.parent).as_posix(), (1980,1,1,0,0,0))
                    entry.create_system = 3
                    entry.external_attr = (stat.S_IFREG | 0o644) << 16
                    entry.compress_type = zipfile.ZIP_DEFLATED
                    entry.comment = b''; entry.extra = b''
                    archive.writestr(entry, path.read_bytes())
            # Verify file bytes and archive metadata without file hashes.
            with zipfile.ZipFile(pending) as archive:
                for info in archive.infolist():
                    assert info.date_time == (1980,1,1,0,0,0)
                    assert not info.extra and not info.comment and not archive.comment
                    assert archive.read(info) == (Path(temporary)/info.filename).read_bytes()
            pending.replace(output)
        finally:
            if pending.exists():
                pending.unlink()
    return dict(files=len(inventory)+1, bytes=output.stat().st_size, audit='passed',
                archive_metadata='normalized', includes_weights=False, includes_git=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--denylist', type=Path, help='Private identifiers, one per line; keep outside source.')
    args = parser.parse_args()
    denylist = ()
    if args.denylist:
        private = args.denylist.resolve()
        if args.root.resolve() in private.parents:
            parser.error('The private denylist must be outside the source repository.')
        denylist = tuple(line.strip() for line in private.read_text().splitlines()
                         if line.strip() and not line.lstrip().startswith('#'))
    print(json.dumps(build(args.root, args.output, denylist), indent=2))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Audit a source-only release without printing matching personal data or secrets.

Run from any directory. An optional private denylist stays outside the release:
``python sihc/tools/audit_release.py . --denylist /path/to/private-identifiers.txt``.
Each nonempty, non-comment denylist line is a case-insensitive literal. The
scanner does not follow symlinks, execute notebooks, contact services, or hash
files. Its checks supplement human review; they cannot prove anonymity.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Iterable


@dataclass(frozen=True, order=True)
class Finding:
    path: str
    line: int
    rule: str
    message: str


# Patterns intentionally contain no release-specific identity or denylist.
PATTERNS = (
    ("private-path", re.compile(r"(?<![\w./\\])/(?:home|Users|data|mnt|scratch|workspace)/[\w.-]+"),
     "Machine-specific absolute path; use a caller-supplied path."),
    ("private-path", re.compile(r"[A-Za-z]:\\(?:Users|Documents and Settings)\\[^\\\s]+"),
     "Machine-specific absolute path; use a caller-supplied path."),
    ("email", re.compile(r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+"),
     "Email address requires removal or a third-party license review."),
    ("credential", re.compile(r"\b(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
     "Possible authentication token."),
    ("credential", re.compile(r"(?i)\b(?:api[_-]?key|(?:api|access|auth)[_-]?token|secret|password)\s*[=:]\s*['\"][A-Za-z0-9+/=_-]{16,}['\"]"),
     "Possible hard-coded authentication value."),
    ("private-key", re.compile("-----BEGIN " + r"(?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
     "Private key material."),
    ("tracking-account", re.compile(r"https?://(?:www\.)?wandb\.ai/[^\s/'\"<>]+/"),
     "Tracking URL can identify an account or historical run."),
    ("hub-account", re.compile(r"https?://(?:www\.)?(?:huggingface\.co|hf\.co)/(?:datasets/|spaces/)?[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+"),
     "Hub account URL requires a public third-party allowlist or removal."),
    ("tracking-identity", re.compile(r"(?i)\bwandb_(?:entity|id|run_id)['\"]?\s*[:=]\s*['\"](?![\s$<{])[^'\"\n]+['\"]"),
     "Hard-coded tracking account or historical run identity."),
    ("author-metadata", re.compile(r"(?i)\b(?:__author__|__maintainer__|author_email|maintainer_email|authors)\s*[:=]"),
     "Author metadata requires removal or explicit anonymous packaging."),
    ("ssh-remote", re.compile(r"(?:ssh:" + r"//[^\s]+|git@[A-Za-z0-9.-]+:)"),
     "SSH remote may disclose an account or inaccessible private source."),
)

ARTIFACT_SUFFIXES = frozenset({
    ".pt", ".pth", ".ckpt", ".safetensors", ".pkl", ".pickle", ".joblib",
    ".npy", ".npz", ".parquet", ".arrow", ".h5", ".hdf5", ".bin",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".mp4", ".wav",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".whl", ".so", ".pyc",
})
PRIVATE_NAMES = frozenset({
    ".env", ".netrc", ".npmrc", ".pypirc", ".git-credentials", ".gitmodules",
    "id_rsa", "id_ed25519", "wandb", ".ssh", ".ipynb_checkpoints",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
})


def scan_text(path: str, text: str, denylist: Iterable[str],
              allow_url_prefixes: tuple[str, ...] = ()) -> list[Finding]:
    findings = []
    for line_number, line in enumerate(text.splitlines(), 1):
        for rule, pattern, message in PATTERNS:
            for match in pattern.finditer(line):
                if rule == "tracking-identity":
                    value = re.search(r"['\"]([^'\"]+)['\"]$", match.group())
                    if value and value.group(1) in {"test", "example", "anonymous"}:
                        continue  # Generic fixtures; private denylist checks still run.
                if rule == "hub-account" and any(
                    match.group().startswith(prefix) for prefix in allow_url_prefixes
                ):
                    continue
                findings.append(Finding(path, line_number, rule, message))
                break
        folded = line.casefold()
        for number, identifier in enumerate(denylist, 1):
            if identifier.casefold() in folded:
                findings.append(Finding(path, line_number, f"private-denylist-{number:03}",
                                        "Private denylist entry matched; value withheld."))
    return findings


def audit_notebook(path: str, content: str) -> list[Finding]:
    findings = []
    try:
        notebook = json.loads(content)
        if not isinstance(notebook, dict) or not isinstance(notebook.get("cells"), list):
            raise ValueError("Invalid notebook structure")
        if set(notebook.get("metadata", {})) - {"kernelspec", "language_info"}:
            findings.append(Finding(path, 0, "notebook-metadata",
                                    "Notebook contains nonessential metadata."))
        for cell in notebook["cells"]:
            if cell.get("outputs") or cell.get("execution_count") is not None:
                findings.append(Finding(path, 0, "notebook-output",
                                        "Notebook must have empty outputs and execution counts."))
            if cell.get("metadata") or cell.get("attachments"):
                findings.append(Finding(path, 0, "notebook-metadata",
                                        "Cell metadata or attachments require removal."))
    except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
        findings.append(Finding(path, 0, "invalid-notebook", "Cannot inspect notebook structure."))
    return findings


def audit_git(root: Path, denylist: tuple[str, ...]) -> list[Finding]:
    """Inspect actual Git metadata; no remote access, hashing, or Git mutation."""
    git_dir = root / ".git"
    if not git_dir.exists():
        return []
    if git_dir.is_symlink() or not git_dir.is_dir():
        return [Finding(".git", 0, "git-metadata", "Release must not point to an external Git worktree.")]
    findings = []
    objects = git_dir / "objects"
    if objects.exists() and any(item.is_file() for item in objects.rglob("*")):
        findings.append(Finding(".git/objects", 0, "git-objects",
                                "Git objects can retain deleted private history; export source files only."))
    config = git_dir / "config"
    if config.exists():
        content = config.read_text(encoding="utf-8", errors="replace")
        findings.extend(scan_text(".git/config", content, denylist))
        if re.search(r"(?m)^\s*\[remote\s", content):
            findings.append(Finding(".git/config", 0, "git-remote",
                                    "Inspect and remove account-bearing remotes before anonymous handoff."))
        if re.search(r"(?m)^\s*\[user\]", content):
            findings.append(Finding(".git/config", 0, "git-identity",
                                    "Local author configuration requires review."))
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "log", "--all", "--format=%an%n%ae%n%cn%n%ce%n%B"],
            capture_output=True, text=True, check=False, timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return findings + [Finding(".git", 0, "git-unchecked", "Could not inspect Git history.")]
    if result.returncode == 0 and result.stdout.strip():
        findings.extend(scan_text(".git/history", result.stdout, denylist))
        findings.append(Finding(".git/history", 0, "git-history",
                                "History exists; distribute a source export for anonymous review."))
    elif result.returncode not in (0, 128):
        findings.append(Finding(".git", 0, "git-unchecked", "Could not inspect Git history."))
    for path in (git_dir / "logs", git_dir / "hooks"):
        if not path.exists():
            continue
        for item in path.rglob("*"):
            if item.is_file() and (path.name != "hooks" or not item.name.endswith(".sample")):
                content = item.read_text(encoding="utf-8", errors="replace")
                findings.extend(scan_text(item.relative_to(root).as_posix(), content, denylist))
    return findings


def audit_tree(root: Path, *, denylist: tuple[str, ...] = (),
               allow_url_prefixes: tuple[str, ...] = (), max_bytes: int = 2_000_000) -> list[Finding]:
    findings = []
    def walk_error(error: OSError) -> None:
        findings.append(Finding(".", 0, "unreadable-directory", "Could not inspect part of the release tree."))

    for directory, dirnames, filenames in os.walk(root, followlinks=False, onerror=walk_error):
        current = Path(directory)
        for name in list(dirnames):
            path = current / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                findings.append(Finding(relative, 0, "symlink", "Symlink may expose external or private files."))
                dirnames.remove(name)
            elif name == ".git":
                dirnames.remove(name)
            elif name in PRIVATE_NAMES or name.endswith(".egg-info"):
                findings.append(Finding(relative, 0, "generated-or-private", "Exclude local metadata and generated artifacts."))
                dirnames.remove(name)
            findings.extend(scan_text(relative, relative, denylist))
        for name in filenames:
            path = current / name
            relative = path.relative_to(root).as_posix()
            findings.extend(scan_text(relative, relative, denylist))
            if path.is_symlink():
                findings.append(Finding(relative, 0, "symlink", "Symlink may expose external or private files."))
                continue
            if name == ".git":
                continue  # audit_git handles worktree pointers.
            if not path.is_file():
                findings.append(Finding(relative, 0, "special-file", "Release contains a nonregular file."))
                continue
            if name in PRIVATE_NAMES or name.endswith((".pem", ".key")):
                findings.append(Finding(relative, 0, "private-file", "Private configuration or key-like file."))
            if path.suffix.casefold() in ARTIFACT_SUFFIXES:
                findings.append(Finding(relative, 0, "artifact", "Source release contains an unreviewed binary/artifact."))
                continue
            if path.stat().st_size > max_bytes:
                findings.append(Finding(relative, 0, "large-file", "File exceeds the source review size limit."))
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                findings.append(Finding(relative, 0, "unreadable-file", "Cannot inspect file as UTF-8 source."))
                continue
            if "\x00" in content:
                findings.append(Finding(relative, 0, "binary-file", "File contains binary data."))
                continue
            findings.extend(scan_text(relative, content, denylist, allow_url_prefixes))
            if path.suffix.casefold() == ".ipynb":
                findings.extend(audit_notebook(relative, content))
    findings.extend(audit_git(root, denylist))
    return sorted(set(findings))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path(__file__).resolve().parents[2])
    parser.add_argument("--denylist", type=Path, help="Private file outside the release; one literal identifier per line.")
    parser.add_argument("--allow-url-prefix", action="append", default=[],
                        help="Explicitly reviewed third-party Hub URL prefix, including its trailing slash.")
    parser.add_argument("--json", action="store_true", help="Emit structured findings, without matching values.")
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        parser.error("root must be an existing directory")
    denylist = ()
    if args.denylist:
        private_path = args.denylist.resolve()
        if private_path == root or root in private_path.parents:
            parser.error("keep the private denylist outside the release")
        denylist = tuple(line.strip() for line in private_path.read_text(encoding="utf-8").splitlines()
                         if line.strip() and not line.lstrip().startswith("#"))
    if any(not value.endswith("/") for value in args.allow_url_prefix):
        parser.error("allowed URL prefixes must end in '/' to avoid account-prefix collisions")
    findings = audit_tree(root, denylist=denylist, allow_url_prefixes=tuple(args.allow_url_prefix))
    if args.json:
        print(json.dumps({"ok": not findings, "findings": [asdict(item) for item in findings]}, indent=2))
    else:
        for item in findings:
            print(f"{item.path}:{item.line}: {item.rule}: {item.message}")
        print(f"{'PASS' if not findings else 'FAIL'}: {len(findings)} finding(s)")
    return int(bool(findings))


if __name__ == "__main__":
    raise SystemExit(main())

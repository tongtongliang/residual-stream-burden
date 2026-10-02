"""The release audit must detect leaks without echoing the matched values."""

import importlib.util
import json
from pathlib import Path
import sys


PATH = Path(__file__).resolve().parents[2] / "sihc" / "tools" / "audit_release.py"
SPEC = importlib.util.spec_from_file_location("release_audit", PATH)
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def rules(root, **kwargs):
    return {finding.rule for finding in audit.audit_tree(root, **kwargs)}


def test_portable_source_passes(tmp_path):
    (tmp_path / "module.py").write_text("from pathlib import Path\nDATA = Path('datasets')\n")
    assert not audit.audit_tree(tmp_path)


def test_paths_accounts_tokens_and_private_denylist(tmp_path):
    private_name = "example" + "researcher"
    token = "hf_" + "a" * 25
    email = "example" + "@" + "example.org"
    content = "\n".join([
        "/" + "home/" + private_name + "/model.py", token, email,
        "https://" + "wandb.ai/" + private_name + "/project",
        "https://" + "huggingface.co/" + private_name + "/checkpoint",
    ])
    (tmp_path / "source.txt").write_text(content)
    found = audit.audit_tree(tmp_path, denylist=(private_name.upper(),))
    assert {item.rule for item in found} >= {
        "private-path", "credential", "email", "tracking-account", "hub-account", "private-denylist-001",
    }
    assert private_name not in repr(found)
    assert token not in repr(found)
    assert email not in repr(found)


def test_symlinks_never_followed_and_artifacts_fail(tmp_path):
    (tmp_path / "outside").symlink_to(tmp_path.parent, target_is_directory=True)
    (tmp_path / "weights.pt").write_bytes(b"not a checkpoint")
    assert {"symlink", "artifact"} <= rules(tmp_path)


def test_notebook_output_and_metadata_fail(tmp_path):
    notebook = {"metadata": {"author": "example"}, "cells": [
        {"cell_type": "code", "execution_count": 1, "outputs": [{"text": "machine information"}],
         "metadata": {"execution": "machine timestamp"}, "source": ["1 + 1"]},
    ]}
    (tmp_path / "analysis.ipynb").write_text(json.dumps(notebook))
    assert {"notebook-metadata", "notebook-output"} <= rules(tmp_path)


def test_stripped_notebook_passes(tmp_path):
    notebook = {"metadata": {"language_info": {"name": "python"}}, "cells": [
        {"cell_type": "code", "execution_count": None, "outputs": [], "metadata": {}, "source": ["1 + 1"]},
    ]}
    (tmp_path / "analysis.ipynb").write_text(json.dumps(notebook))
    assert not audit.audit_tree(tmp_path)


def test_hub_allowlist_has_explicit_account_boundary(tmp_path):
    account_url = "https://" + "huggingface.co/public-vendor/"
    (tmp_path / "README.md").write_text(account_url + "public-model")
    assert "hub-account" in rules(tmp_path)
    assert not audit.audit_tree(tmp_path, allow_url_prefixes=(account_url,))


def test_filename_denylist_and_local_metadata_fail(tmp_path):
    (tmp_path / "private-person.txt").write_text("clean content")
    (tmp_path / "wandb").mkdir()
    assert {"private-denylist-001", "generated-or-private"} <= rules(tmp_path, denylist=("private-person",))


def test_binary_text_extension_fails(tmp_path):
    (tmp_path / "misnamed.txt").write_bytes(b"source\x00private")
    assert "binary-file" in rules(tmp_path)


def test_author_metadata_requires_explicit_review(tmp_path):
    field = "__" + "author__"
    (tmp_path / "module.py").write_text(field + " = 'example researcher'\n")
    assert "author-metadata" in rules(tmp_path)


def test_external_git_worktree_pointer_fails(tmp_path):
    (tmp_path / ".git").write_text("gitdir: ../another-checkout/.git/worktrees/export\n")
    assert "git-metadata" in rules(tmp_path)


def test_tracking_placeholders_are_generic_but_denylist_still_applies(tmp_path):
    field = "wandb_" + "run_id"
    (tmp_path / "fixture.py").write_text(field + "='test'\n")
    assert not audit.audit_tree(tmp_path)
    assert "private-denylist-001" in rules(tmp_path, denylist=("test",))
    (tmp_path / "fixture.py").write_text(field + "='historical-identity'\n")
    assert "tracking-identity" in rules(tmp_path)


def test_unreachable_git_objects_are_not_ignored(tmp_path):
    objects = tmp_path / ".git" / "objects" / "pack"
    objects.mkdir(parents=True)
    (objects / "unreferenced.pack").write_bytes(b"unreachable private history")
    assert "git-objects" in rules(tmp_path)


def test_relative_markdown_links_are_not_machine_paths(tmp_path):
    path = tmp_path / "README.md"
    path.write_text("[data](../data/README.md)\n[local](docs/data/README.md)\n")
    assert not audit.audit_tree(tmp_path)
    path.write_text("source = '" + "/" + "data/example-source/images'\n")
    assert "private-path" in rules(tmp_path)

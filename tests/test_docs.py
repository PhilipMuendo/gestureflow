"""Documentation guards: generated reference is current, links resolve, named identifiers exist."""

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOC_FILES = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
sys.path.insert(0, str(ROOT / "tools"))


def test_configuration_reference_is_up_to_date():
    import gen_config_docs

    current = (ROOT / "docs" / "configuration.md").read_text(encoding="utf-8")
    assert current == gen_config_docs.render(), "run: python tools/gen_config_docs.py"


def test_every_config_field_is_documented():
    import gen_config_docs

    text = gen_config_docs.render()
    assert "|  |" not in text, "a config field has no comment in config.py, so it has no description in the docs"


@pytest.mark.parametrize("doc", DOC_FILES, ids=lambda p: p.name)
def test_relative_links_and_images_resolve(doc):
    text = doc.read_text(encoding="utf-8")
    for target in re.findall(r"\]\(([^)\s]+)\)", text):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        path = (doc.parent / target.split("#")[0]).resolve()
        assert path.exists(), f"{doc.name}: broken link -> {target}"


@pytest.mark.parametrize("doc", DOC_FILES, ids=lambda p: p.name)
def test_internal_anchors_exist(doc):
    """`[text](#anchor)` and `[text](file.md#anchor)` must point at a real heading."""

    def slug(heading: str) -> str:
        s = re.sub(r"[`*_]", "", heading.strip().lower())
        s = re.sub(r"[^\w\s-]", "", s)
        return re.sub(r"\s", "-", s)

    for target in re.findall(r"\]\(([^)\s]*#[^)\s]+)\)", doc.read_text(encoding="utf-8")):
        file_part, anchor = target.split("#", 1)
        page = doc if not file_part else (doc.parent / file_part)
        if not page.exists() or page.suffix != ".md":
            continue
        headings = [slug(h) for h in re.findall(r"^#{1,6}\s+(.*)$", page.read_text(encoding="utf-8"), re.M)]
        assert anchor in headings, f"{doc.name}: no heading for #{anchor} in {page.name}"


def test_identifiers_named_in_the_technical_docs_exist_in_the_code():
    """Catches stale names: any `snake_case_token` in backticks must still exist somewhere in the source."""
    source = "\n".join(p.read_text(encoding="utf-8") for p in ROOT.rglob("*.py") if ".venv" not in p.parts)
    known = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", source))
    missing = []
    for doc in (ROOT / "docs").glob("*.md"):
        if doc.name == "configuration.md":
            continue
        for token in re.findall(r"`([a-z][a-z0-9]*(?:_[a-z0-9]+)+)`", doc.read_text(encoding="utf-8")):
            if token not in known:
                missing.append(f"{doc.name}: {token}")
    assert not missing, "identifiers in the docs that no longer exist in the code:\n" + "\n".join(sorted(set(missing)))

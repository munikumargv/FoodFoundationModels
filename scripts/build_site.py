"""Build the static GitHub Pages site into _site/.

The page runs the nutrition gap analyzer in the browser with Pyodide. YAML
files are converted to JSON (valid YAML) and a tiny `yaml` shim is bundled, so
no extra Pyodide packages are needed.

    python scripts/build_site.py [--out _site]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
YAML_SHIM = '''"""Minimal yaml shim for the browser bundle: data files are JSON (a subset of YAML)."""
import json


def safe_load(stream):
    return json.load(stream) if hasattr(stream, "read") else json.loads(stream)
'''
# Only what the built-in engine needs (no torch-dependent modules).
PACKAGE_FILES = ["__init__.py", "knowledge.py", "profile.py", "calculators.py", "labs.py", "gaps.py"]


def build(out: Path) -> Path:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    with zipfile.ZipFile(out / "foodfm_bundle.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for name in PACKAGE_FILES:
            z.write(ROOT / "src" / "foodfm" / name, f"src/foodfm/{name}")
        for folder in ("knowledge", "configs"):
            for f in sorted((ROOT / folder).glob("*.yaml")):
                data = yaml.safe_load(f.read_text(encoding="utf-8"))
                z.writestr(f"{folder}/{f.name}", json.dumps(data, ensure_ascii=False))
        z.writestr("yaml.py", YAML_SHIM)
        z.write(ROOT / "web" / "web_api.py", "web_api.py")
    shutil.copy(ROOT / "web" / "index.html", out / "index.html")
    # Pre-exported so the UI renders immediately while the Python engine loads.
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "web")]
    import web_api

    (out / "bootstrap.json").write_text(web_api.bootstrap(), encoding="utf-8")
    (out / ".nojekyll").touch()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "_site"))
    print(f"Site built in {build(Path(ap.parse_args().out))}")


if __name__ == "__main__":
    main()

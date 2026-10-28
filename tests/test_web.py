import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_site import build  # noqa: E402


def test_build_site_and_run_bundle_without_pyyaml(tmp_path):
    site = build(tmp_path / "site")
    assert (site / "index.html").exists() and (site / ".nojekyll").exists()
    app = tmp_path / "app"
    with zipfile.ZipFile(site / "foodfm_bundle.zip") as z:
        assert not any(n.endswith(("train.py", "infer.py")) for n in z.namelist())
        z.extractall(app)

    # Run the bundle in a fresh interpreter exactly as the browser does: the
    # bundled yaml shim must be used and the JSON-converted data must load.
    code = f"""
import json, os, sys
sys.path[:0] = [{str(app)!r}, {str(app / "src")!r}]
os.environ["FOODFM_KNOWLEDGE_DIR"] = {str(app / "knowledge")!r}
os.environ["FOODFM_CONFIG_DIR"] = {str(app / "configs")!r}
import yaml, web_api
assert yaml.__file__.startswith({str(app)!r}), yaml.__file__
opts = json.loads(web_api.options())
assert len(opts["countries"]) == 25
res = json.loads(web_api.recommend(json.dumps({{"country": "kenya", "age": 30, "sex": "male"}})))
assert res["errors"]
res = json.loads(web_api.recommend(json.dumps({{"country": "india", "age": 30, "sex": "female", "weight_kg": 55, "height_cm": 160}}), True))
assert "Nutrition plan for a 30-year-old female in India" in res["plan"] and "PROFILE TARGETS" in res["prompt"]
print("ok")
"""
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"

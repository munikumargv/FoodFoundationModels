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
    assert (site / "index.html").exists() and (site / ".nojekyll").exists() and (site / "bootstrap.json").exists()
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
boot = json.loads(web_api.bootstrap())
assert len(boot["countries"]) == 25 and boot["personas"]
p = boot["personas"][0]
res = json.loads(web_api.analyze(json.dumps({{"profile": p["profile"], "meals": p["meals"], "lab_text": p["lab_report"]}})))
assert res["gaps"] and res["supplements"] and res["food_recommendations"], res.keys()
assert json.loads(web_api.analyze(json.dumps({{"profile": {{"country": "atlantis", "age": 30, "sex": "male"}}, "meals": []}})))["errors"]
assert json.loads(web_api.detect("dal_rice.jpg", "india"))[0]["food"] == "dal"
assert json.loads(web_api.parse_labs("Ferritin 9 ng/mL"))["ferritin"]["value"] == 9
print("ok")
"""
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"

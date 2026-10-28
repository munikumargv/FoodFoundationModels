# NutriScan — nutrition gap analyzer (FoodFM)

**Snap your meals, add your lab report, and see your nutrition gaps.**

NutriScan takes a person's **country**, **food photos** and, optionally, a **lab report**. It produces:

1. **Clear nutrition gaps**: daily intake of 10 nutrients compared with the RDA for that person's age, sex and
   life stage (WHO / NASEM), plus WHO limits for sodium, free sugars and saturated fat. Lab results confirm
   or upgrade each gap; for example, low ferritin turns an iron shortfall into a confirmed deficiency.
2. **Supplements required**, each with a dose, a priority, the reason (lab value or diet shortfall), tips and
   safety cautions.
3. **Food recommendations from the person's country** that close the gaps and respect their diet
   (vegetarian, vegan, halal, kosher, Jain…) and allergies, plus healthier local swaps.

> ⚠️ NutriScan is a screening and education tool, not medical advice. Nutrient values are approximate, and
> supplement suggestions should be confirmed with a doctor or registered dietitian.

**Live demo:** `https://<user>.github.io/FoodFoundationModels/`. It opens with demo personas, so it can be
shown without real photos or reports.

| Persona | Story | What the demo shows |
|---|---|---|
| 👩🏽 Priya 🇮🇳 | 32, vegetarian, tired | Iron, B12 and vitamin D deficiency confirmed by labs → iron, D3 and B12 supplements; palak, chana, rajma |
| 👨🏼 James 🇺🇸 | 45, desk job, fast food | Sodium, sugar and saturated fat over the limit; prediabetes HbA1c; fibre and vitamin C gaps |
| 🤰🏿 Amara 🇳🇬 | 27, pregnant (2nd trimester) | Anaemia, folate and calcium gaps; pregnancy-specific iron and folic acid; ewa, dagaa, ugwu |
| 👴🏻 Hiroshi 🇯🇵 | 74, retired | Vitamin D deficiency; B12 intake fine but blood level low (an absorption flag); salt from miso/ramen |

## How it works

```
 📸 meal photos ──► food detection ──► foods + portions ──┐
                    (vision model / file name / manual)    ├─► daily intake ─┐
 🧑 profile + country ──► RDA targets (age, sex, pregnancy, diet) ───────────┼─► gap analysis ─► 🎯 gaps
 🧪 lab report (PDF/TXT/CSV) ──► parser ──► marker status ───────────────────┘          │      💊 supplements
                                                                                         └────► 🥗 local foods
```

| Step | Logic | Code / data |
|---|---|---|
| Food detection | A photo's label (file name today, vision-model caption when connected) is matched against a database of 97 foods and dishes from all 25 countries; the user confirms foods and portions | `gaps.detect_foods`, `knowledge/foods.yaml` |
| Intake | Nutrients per serving × portions, summed over the day | `gaps.daily_intake` |
| Targets | NASEM RDA/AI by life stage and sex, energy from Mifflin-St Jeor / Schofield × activity, protein by g/kg, ×1.8 iron for plant-based diets, WHO sodium/sugar/saturated-fat limits | `calculators.py`, `knowledge/standards.yaml` |
| Gap status | < 50% of RDA deficient · 50–80% low · 80–100% borderline · ≥ 100% adequate | `gaps.analyze` |
| Labs | 13 markers (haemoglobin, ferritin, vitamin D, B12, folate, zinc, calcium, albumin, HbA1c, glucose, LDL, triglycerides, TSH) with unit conversion and sex/pregnancy-aware cut-offs. Labs escalate gaps, and a "diet fine but blood low" result adds an absorption flag | `labs.py`, `knowledge/labs.yaml` |
| Supplements | Recommended on a deficient/low lab result, on a large diet shortfall, or always (B12 for vegans and over-50s, folic acid and iron in pregnancy, D3 at 71+). Blocked on unsafe results (high ferritin → no iron; high calcium → no calcium or D). Children get "ask your paediatrician" in place of a dose | `gaps._supplements` |
| Food picks | The highest nutrient-per-serving foods for each gap, local foods weighted 1.5×, filtered by diet and allergies, excluding salty, sugary or fatty items | `gaps._food_recommendations` |

## Run it

**Web app** (the same app GitHub Pages serves):

```bash
python scripts/build_site.py && python -m http.server -d _site 8000
```

Then open http://localhost:8000. The analysis engine is this repo's Python code, running in the browser with
[Pyodide](https://pyodide.org). Nothing a user uploads leaves their device.

**Command line:**

```bash
pip install -e ".[dev]"
```

```bash
foodfm analyze --persona priya
```

```bash
foodfm analyze --input my_day.json --labs my_report.txt --json
```

`my_day.json` looks like this:

```json
{"profile": {"country": "india", "age": 32, "sex": "female", "weight_kg": 58, "height_cm": 158,
             "activity_level": "light", "dietary_pattern": "vegetarian_lacto_ovo"},
 "meals": [{"name": "Lunch", "items": [{"food": "dal", "servings": 1}, {"food": "white_rice", "servings": 1.5}]}]}
```

## Real photo recognition (vision model)

The static demo can't run a vision model, so for uploaded photos it matches foods from the **file name**
(for example `dal_rice_lunch.jpg`), and the user can tag foods by hand. To recognise real photos, run any
vision model behind an HTTP endpoint and pass it to the page:

```
https://<user>.github.io/FoodFoundationModels/?vision=https://your-server/detect
```

The page `POST`s each photo as multipart (`image`, `country`). The endpoint returns either
`{"labels": ["dal", "rice", "curd"]}`, which is mapped onto the food database, or
`{"items": [{"food": "dal", "servings": 1.5}]}`. Good open choices include **Qwen2.5-VL** or **LLaVA**
prompted to list the dishes and portions, or a Food-101 classifier. The repo's QLoRA pipeline (below) can
fine-tune the language side.

## Lab reports

- **Text-based PDF, TXT, CSV**: parsed automatically (pdf.js). For example `Hemoglobin 10.8 g/dL`,
  `Vitamin D (25-OH) 45 nmol/L` (converted to 18 ng/mL) and `HbA1c, 6.2, %`.
- **Photos of reports**: type the values into the form. OCR would need the vision backend.
- Every value can be edited before analysis.

## Deploying on GitHub

| What | Workflow |
|---|---|
| Tests on every push and pull request (Python 3.10, 3.12, 3.13) | `.github/workflows/ci.yml` |
| Web app on GitHub Pages | `.github/workflows/pages.yml` |

One-time setup: **Settings → Pages → Build and deployment → Source: GitHub Actions**. After that, every
push to the default branch that changes the engine, data or page redeploys it.

## Customising the data

Everything is plain YAML in `knowledge/`:

| File | Contents |
|---|---|
| `foods.yaml` | food composition per serving, diet tag, allergens, countries, photo keywords |
| `labs.yaml` | lab markers (aliases, units, cut-offs) and supplement rules (doses, triggers, cautions) |
| `standards.yaml` | RDA/AI tables, WHO limits, energy method |
| `countries.yaml` | food habits, guidelines and swaps for the 25 countries |
| `demo_personas.yaml` | the demo personas (default values) |
| `age_groups.yaml`, `factors.yaml` | life stages and personal factors |

Run `pytest` after editing; it checks the data is consistent. For production, swap `foods.yaml` for a
validated national food composition database.

## Also included: grounded open-LLM pipeline (optional)

The repo still contains the toolkit for fine-tuning an open LLM (Qwen, Llama, Mistral, Gemma, Phi) on this
knowledge base with QLoRA. It isn't needed for the analyzer, but it can generate narrative explanations or
serve as the language side of a vision-language model.

```bash
pip install -e ".[train]"
```

```bash
foodfm build-dataset --n-profiles 3000
```

```bash
foodfm train --model qwen2.5-7b
```

```bash
foodfm evaluate --adapter outputs/foodfm-lora
```

A 7–8B model needs about 16 GB of VRAM. `scripts/smoke_train_tiny.py` checks the pipeline offline with a tiny
random model. Model choices are in `configs/models.yaml` and hyper-parameters in `configs/train.yaml`.

## Project layout

```
web/          NutriScan app (index.html) and its Python bridge (web_api.py)
src/foodfm/   gaps.py (analyzer), labs.py (lab parsing), calculators.py (RDA/energy), knowledge.py, profile.py,
              cli.py, and the optional LLM pipeline (dataset, train, infer, evaluate, grounding, retrieval)
knowledge/    foods, labs and supplements, standards, countries, demo personas (YAML)
scripts/      build_site.py (Pages bundle), smoke_train_tiny.py
configs/      open-model choices and QLoRA settings
tests/        pytest suite
```

## Limitations

- Portions come from photos and standard servings, so intake is an estimate. One day of meals may not be
  typical, so analyse several days for a better picture.
- Food values are approximate reference values, and recipes vary. Iodine, potassium, omega-3 and vitamin K
  are not yet tracked.
- Lab cut-offs vary between laboratories and guidelines; the person's own value is always shown.
- Clinical cases (pregnancy, children, kidney disease, eating disorders) get general guidance and are referred
  to a clinician.

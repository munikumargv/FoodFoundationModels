# FoodFM — a nutrition-grounded open LLM

FoodFM takes an **open-weight LLM** (Qwen, Llama, Mistral, Gemma, Phi — your choice) and customises it so it
answers **only nutrition questions**, using:

- **Nutrition standards**: WHO population goals, NASEM Dietary Reference Intakes (RDA/AI/AMDR), FAO/WHO/UNU
  energy equations, ESPEN/PROT-AGE protein guidance for older adults.
- **Food habits of the top 25 countries** (by population): staples, typical meals, national dietary
  guideline, healthy traditions, public-health concerns, cultural/religious practices, local sources of
  protein/iron/calcium/B12, and culturally familiar healthier swaps.
- **Age**: infant, toddler, child, pre-teen, adolescent, adult, older adult, elderly, plus pregnancy (by
  trimester) and breastfeeding.
- **Other factors**: sex, activity level, dietary/religious pattern (vegetarian, vegan, halal, kosher, Jain,
  Hindu, Buddhist, keto…), medical conditions (diabetes, hypertension, CKD, celiac, anaemia, PCOS, eating-disorder
  history…), allergies, goal (weight loss/gain, muscle), budget, climate, lifestyle (smoking, alcohol, shift work),
  and diet–drug interaction warnings.
- **Interactive selection prompts**: the CLI asks you to pick the model, country, age, and every factor
  from numbered menus.

> ⚠️ FoodFM gives general, standards-based information. It is not a substitute for a doctor or registered
> dietitian. Review the knowledge base against the latest guideline editions before any real-world use.

## How the "nutrition-only" grounding works

You can't make a pretrained LLM *forget* everything else, so FoodFM gets the same effect with four layers:

| Layer | What it does | Where |
|---|---|---|
| 1. Curated knowledge base | The only source of truth: standards, 25 countries, age groups, factors (YAML) | `knowledge/` |
| 2. Deterministic calculator | Computes energy (Mifflin-St Jeor / Schofield × PAL), protein, macros, sodium and micronutrient targets so the model never invents numbers | `src/foodfm/calculators.py` |
| 3. Fine-tuning (QLoRA) | Trains on an instruction dataset generated from the knowledge base: personalised plans, standards Q&A, country habits, age/factor advice, safety cases, and **refusals of off-topic questions** | `src/foodfm/dataset.py`, `src/foodfm/train.py` |
| 4. Retrieval grounding (RAG) | At inference, the profile targets and the top BM25-matched knowledge chunks go into the prompt, and a system prompt limits the model to nutrition | `src/foodfm/grounding.py`, `src/foodfm/retrieval.py` |

```
 knowledge/*.yaml ──► dataset.py ──► data/train.jsonl ──► train.py (QLoRA) ──► outputs/foodfm-lora
        │                                                                          │
        ├──► calculators.py ─┐                                                     ▼
        └──► retrieval.py ───┴──► grounding.py (system prompt + profile + facts) ──► infer.py ──► answer
                                                         ▲
                                            cli.py selection prompts
```

## Quick start

```bash
pip install -e ".[dev]"
```

```bash
foodfm
```

That opens the main menu:

```
What would you like to do?
   1. Get a personalised nutrition recommendation
   2. Chat with the model (nutrition questions)
   3. Build the training dataset from the knowledge base
   4. Fine-tune (QLoRA) an open model
   5. Evaluate a model on the held-out set
   6. List available open models
   7. Exit
```

**Recommendation questionnaire**: country (25 options) → age (`34` or `8 months`) → sex → weight/height
(optional) → pregnancy/lactation (when it applies) → activity → dietary pattern → medical conditions
(multi-select) → allergies (multi-select) → goal → budget → climate → lifestyle → **answer engine**:

1. **Built-in grounded engine**: instant, no GPU; renders a plan straight from the standards and country rules.
2. **Open LLM + retrieval**: pick a base model from `configs/models.yaml`, optionally with your fine-tuned adapter.

You can save the profile to JSON and skip the questionnaire next time:

```bash
foodfm recommend --profile examples/profile_india_pregnant_vegetarian.json --engine builtin --show-prompt
```

## Fine-tuning your own model

The default is QLoRA: 4-bit base, LoRA r=16 on all linear layers. A 7–8B model needs about 16 GB of VRAM
(a T4/A10/L4 or an RTX 4090). `phi3.5-mini` fits in about 10 GB.

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

```bash
foodfm chat --adapter outputs/foodfm-lora --profile examples/profile_usa_adult_diabetes.json
```

- `--model` takes a key from `configs/models.yaml` (`qwen2.5-7b` (default), `qwen3-8b`, `llama3.1-8b`,
  `mistral-7b`, `gemma2-9b`, `phi3.5-mini`, `qwen2.5-0.5b`) or any Hugging Face model id / local path.
- Gated models (Llama, Gemma) need `huggingface-cli login` and licence acceptance first.
- Hyper-parameters live in `configs/train.yaml`. Override them with `--epochs`, `--max-steps`, `--output-dir`, `--no-4bit`.
- Loss is computed only on the assistant reply. Templates without a system role (Gemma) get the system
  prompt merged into the first user turn automatically.

The dataset at the defaults is about 4.7k examples (split into train/eval so no question+answer pair is in both):

| kind | content |
|---|---|
| `profile_plan` | random but valid profiles (all 25 countries, every age band, factors combined) → full grounded plan |
| `standard_fact` | micronutrient values for each life stage and sex, WHO goals, AMDR |
| `country_habit` | staples, concerns, guideline, traditions, swaps, cultural practices, local nutrient sources |
| `age_group` / `factor` | life-stage priorities; adjustments for each condition, allergy, diet, goal and so on |
| `safety` | child weight loss, very-low-calorie diets, honey for infants, CKD, pregnancy, warfarin |
| `out_of_scope` | coding, trivia, politics, drug dosing… → polite refusal (upsampled) |

**Evaluation** (`foodfm evaluate`) reports, per kind: `number_recall` (were the standard values reproduced?),
`token_f1`, `refusal_acc` on off-topic questions, and `false_refusal` on in-scope ones. Run it on the base
model and on the adapter to measure what fine-tuning added.

## Customising the knowledge

Everything the model knows is in plain YAML:

- `knowledge/standards.yaml`: WHO goals, AMDR, protein, micronutrient tables, energy method
- `knowledge/countries.yaml`: the 25 countries (edit the list to rank "top 25" by GDP or anything else)
- `knowledge/age_groups.yaml`: life stages, pregnancy and lactation
- `knowledge/factors.yaml`: every selectable factor with its adjustments and safety notes

After editing, run `pytest` (it checks completeness and catches YAML pitfalls), then rebuild the dataset and
retrain. Changes to the knowledge base reach the built-in engine and retrieval right away; the fine-tuned
weights pick them up after retraining.

## Project layout

```
configs/      models.yaml (selectable open models), train.yaml (QLoRA hyper-parameters)
knowledge/    curated nutrition knowledge base (YAML)
examples/     sample profiles
src/foodfm/   knowledge, profile, calculators, retrieval, grounding, dataset, chat_format, train, infer, evaluate, cli
scripts/      smoke_train_tiny.py: offline end-to-end check of train + infer with a tiny random model
tests/        pytest suite
```

## Limitations

- Reference values describe healthy populations. Clinical nutrition (CKD, eating disorders, severe
  malnutrition, paediatrics) is flagged and referred to a clinician on purpose.
- Children's BMI needs WHO BMI-for-age z-scores, and energy for under-3s follows growth and appetite;
  neither is computed.
- The country profiles summarise typical national patterns. Regional, urban/rural and individual diets vary widely.
- Guideline editions change (for example the Dietary Guidelines for Americans 2025–2030). Check
  `last_reviewed` in each file.

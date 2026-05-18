---
name: prepare-datasets
description: Prepare verified raw/probed datasets into Hugging Face-ready train/eval artifacts for benchmark environments. Use when asked to build or refactor data prep notebooks or scripts, create splits, audits, dataset cards, media excerpts, validation outputs, or publishable dataset schemas before environment integration.
---

# Prepare Datasets

## Goal
Turn a verified source/probe into a scientifically useful dataset artifact. Optimize first for a trustworthy test set; when viable, also create train data that supports general learning and transfer.

## Start
1. Before designing or editing prep code, inspect existing data-prep notebooks/scripts in the repo for grounding. Read enough of most/all relevant notebooks, not just one. Inspect input cells and, when present, executed outputs/previews/errors/artifacts. Learn the local style, cell/script flow, `#@param` configuration conventions, audit patterns, split conventions, HF schema style, dataset-card style, separate push-cell pattern, and how nearby envs expect source-shaped data.
2. Read the probe report/artifacts, source documentation, and relevant env code before designing prep.
3. Match existing repo patterns when they fit so new prep feels native to the lab. Reuse conventions for setup, user-configurable top-panel knobs, outputs, visual audits, split reports, HF previews, dataset-card scaffold cells, separate optional Hub push cells, and review checklists; do not cargo-cult old structure when the new dataset needs a different shape.
4. Respect the user’s preferred artifact form: notebook, `.py` script, or both. Direct `.ipynb` edits are okay when the user chose notebooks or the workspace convention clearly points there.
5. Do not write prep code blind. For manageable data, run local exploration first and know what each notebook cell or script step should output before encoding it. Use current outputs and file checks to determine which branch will actually run in the user's environment; do not leave it as a vague conditional when the evidence is available.

## Exploration Loop
- Load/sample actual data and inspect schema, labels, IDs/groups, media properties, missingness, and source quirks.
- Build visual audits that are scientifically meaningful for the dataset: label balance, group coverage, quality filters, PCA/embeddings, split balance, and representative examples.
- Form hypotheses about label semantics, leakage, split units, quality thresholds, and model difficulty; test them with code; iterate until the data shape is understood.
- For media, decode/view/listen to examples and record technical metadata. Add excerpts, ceilings, resizing, or opaque filenames when needed for model budgets or leakage control.
- Audit modality claims before publishing. If a source contains audio/video/images incidentally, verify whether that modality is scientifically meaningful for the label or merely noisy/confounded. Drop weak modalities from the public/default dataset when they reduce interpretability; expose them only as explicit ablations when useful.

## Splits
- Avoid naive row-level random splits. Choose split units by contamination risk: animal/subject, owner, farm, session, source file, citation, experimental day, or other source group.
- If true IID stratification is appropriate, use it; otherwise use grouped/heldout splits and search seeds to preserve representative label/task coverage.
- Prefer `train`/`test` when viable. Add `train_raw`/`test_raw` when balanced benchmark views differ from the natural source distribution.
- Test set goal: a representative capability signal where better models should beat weaker ones.
- Train set goal: broad, diverse learning signal that improves transfer, not just memorization of test-like rows.

## HF Dataset Shape
- Publish source-shaped data: labels, media, IDs, safe metadata, attribution, and split membership.
- Do not publish env-shaped `messages`, fixed prompts, or baked instructions unless the user explicitly asks for SFT/chat data. Context rendering belongs in the environment so env args can change modality, instructions, and ablations.
- Do not invent lab-wide or suite-wide row IDs such as `openfarm_id` unless the source itself provides a stable identifier and the user wants it public. Prefer source-native IDs when meaningful, split/group IDs for contamination audits, and opaque media filenames when leakage control needs renamed files.
- Keep lab/env bookkeeping such as `task_id`, prompt variants, or helper IDs out of the public HF rows by default. If a helper column is needed while preparing data, make it local/private or drop it before the DatasetDict preview.
- Keep enough metadata for audits and future ablations. If source filenames or fields encode answers, either rename/exclude them or clearly isolate them from default public rows according to the user’s preference.
- Public filenames should not casually encode labels, source groups, split membership, or answer-bearing clip codes. Use opaque media filenames when source names or derived names could leak.
- Use native HF media features (`Audio`, `Image`, `Video`) when applicable, and verify that files exist and decode before push. For path-backed media, inspect the DatasetDict storage and a forced reload: public rows must contain portable bytes or Hub-resolvable paths, not `/content` paths, repo-local scratch paths, or `bytes=None` surprises.

## Validation
Include the checks that match the dataset:
- Source manifest, dataset audit JSON, visual audit images, split metadata, and dataset card.
- Public schema assertions, forbidden-field checks, media existence/decode checks, duration/size ceilings, and split-overlap checks.
- Simple baselines when possible: majority, shallow lexical/acoustic/feature model, confusion matrix, balanced accuracy, macro F1.
- HF `DatasetDict` preview before publishing. Keep tokens private and make push/visibility an explicit user-controlled step.

## Handoff
Report the output path, public schema, split rationale, verification performed, known limitations, and the next environment/training/eval step.

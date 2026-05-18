---
name: probe-data-sources
description: Find and verify public permissionless datasets for a benchmark, environment, or research domain before data prep. Use when asked to discover dataset sources, probe candidate data, inspect accessibility/license/shape, stream or sample examples, or recommend whether a source is suitable for train/eval in a Prime Lab/OpenFARM-style workflow.
---

# Probe Data Sources

## Goal
Given a benchmark idea or domain, find public datasets that can realistically become train/eval data, then verify what is actually accessible before recommending data prep.

## Workflow
1. Clarify the target only if needed: species/domain, modality, label type, train vs eval need, size budget, and license constraints.
2. Search current web sources and prefer primary pages: dataset hubs, Zenodo/Figshare/Mendeley/OSF, Hugging Face, official project pages, papers, and code repos.
3. Treat a dataset as permissionless only when data is reachable without login, forms, manual approval, click-through legal acceptance, or non-public credentials. Do not accept licenses on the user's behalf.
4. Check license and redistribution/training fit. If terms are missing, contradictory, non-commercial, no-derivatives, no-sharing, or request-only, say so plainly.
5. Verify access instead of trusting descriptions:
   - For HF/parquet/CSV/JSON, stream or load a tiny split/sample when practical and inspect fields.
   - For archives/media, prefer metadata/API manifests, file listings, checksums, and a few small sampled examples before bulk download.
   - For large sources, do a metadata-only probe and estimate size/runtime before downloading.
6. Inspect shape enough to judge benchmark value: modalities, row counts, labels, label balance, subject/source IDs, split candidates, media duration/resolution, missing fields, and obvious leakage risks.
7. Decode, view, or listen to examples when manageable. For media, verify files actually open and record basic technical metadata.
8. Distinguish the source's primary scientific object from incidental modalities. For example, a video dataset may contain audio that is noisy, silent, or irrelevant to labels; do not recommend a modality just because bytes exist.
9. Stop before full data prep unless the user asks to proceed.

## Reporting
Return a compact report with:
- Verdict: use now, promising later, blocked, or reject.
- Source links, DOI/citation when available, license, and access status.
- Verified data shape and sample evidence.
- Benchmark/env tasks it could support.
- Train/eval suitability, split/leakage concerns, modality caveats, and redistribution risks.
- Concrete next step if the user wants to prep it.

## Local Artifacts
Create local probe artifacts only when useful for verification or handoff. Choose the simplest structure for the source; do not copy previous probe layouts by rote. Keep raw downloads small by default, and record any large files intentionally skipped.

## Guardrails
- Do not present paper claims as verified data access.
- Do not use private credentials, scraped login sessions, or accepted legal terms unless the user explicitly handles that access.
- Do not push datasets to Hugging Face or start notebook/env implementation from this skill alone.
- Prefer scientific honesty over optimism: a blocked but excellent dataset is still blocked.

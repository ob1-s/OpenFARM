# Track B — cow locomotion and positive clinical outcomes (source-level checkpoint)

**Research branch:** `research/2026-10-data-prospecting`  
**This report does not promote a source to approved training/evaluation without actual permissions, independent labels, and appropriate heldout identities.**

## A. Russello et al. 2026 — verified real 2D cow gait pose data

- Source: [hrussel/lstm-lameness-detection](https://github.com/hrussel/lstm-lameness-detection), pinned commit `c9e2f6d558912589220d91de985a79851d6e2c97`; paper [DOI 10.1016/j.atech.2026.101831](https://doi.org/10.1016/j.atech.2026.101831).
- **Real source audit:** the entire public repository's pose files and score CSV were cloned read-only and verified by [CI 38826793570](https://github.com/ob1-s/OpenFARM/actions/runs/38826793570) using `scripts/audit_cow_lameness_pose.py`.
- **272** individual trajectory CSV files, **272** score rows, **98** independently identifiable cow IDs, **36,548** source pose frames, all original trajectory/score file names paired; no exact duplicate trajectory files; no missing pose numeric values. Frame sequences range **90–207** frames (median 130).
- Original `hard_vote` scores are **1/2/3/4** with **143/96/20/13** sequences. **Do not relabel as 0–3 without explicitly documenting a reversible mapping to the source rubric.**
- **74/98 cows have repeated clips**, maximum eight clips/cow. **34 cows have varying score categories across clips**, which may reflect temporal changes or label disagreement. Therefore evaluation MUST split by cow ID (and site/session if present), not individual clip. Sample size 272 ≠ 272 independent cattle.
- The repository has **no LICENSE file and no explicit GitHub repo license**. Publicly readable does **not** establish permission for copying to our repository, general ML training, model distribution, or commercial reuse. **Rights clearance is REQUIRED** before promoting to `APPROVED_RESEARCH_TRAIN/EVAL` under our strict policy.
- Label source: the repository points to the authors' 2024 locomotion work. The score CSV provides **hard_vote**, not each original human assessor's vote. Thus it does *not* yet qualify as a calibrated **multi-rater** ordinal benchmark, nor can we infer rater distribution from hard votes alone.
- **Status:** `ACCESS_VERIFIED / TASK_VALIDATION_PENDING / RIGHTS_PENDING`. Potential low-cost structured pose `Score(1–4)` candidate, subject-heldout and rating provenance mandatory.

## B. USP-linked ten-rater top/side cow lameness project — access blocked at actual files

- Primary page: [Animal Welfare Science HUB — Lameness in Dairy Cows](https://animalwelfarehub.org/cows-project/).
- The project states **215 paired observations** comprising 215 top-view + 215 side-view MP4s (**430 videos, not 430 independent cow events**). Ten trained observers scored each pair on DairyNZ **0–3** locomotion rubric. Score aggregation described; severity distribution stated as 32 score-0, 109 score-1, 60 score-2, 14 score-3 paired observations.
- The landing page states a spreadsheet exists with all ten individual ratings, means, SD and coefficient of variation. It also gives an email for 3D source files. But **no video or score-sheet download address could be recovered** from public HTML, iframe, sources or scripts on the project page or the parent repository page.
- **Independently checked** in [CI 38826844109](https://github.com/ob1-s/OpenFARM/actions/runs/38826844109) using `scripts/audit_cow_project_access.py`: the direct-page source is 162.7 KB and lists no linked MP4/XLSX/CSV/media or downloadable source files; the adjacent data-repository page simply links back to the project. The website's description of files is **not** evidence the data can be ingested. The parent site mentions CC BY 4.0 with a warning about third-party rights; **no direct per-media license verified**.
- **Status:** `METADATA_VERIFIED / ACCESS_PENDING`, not an approved training or evaluation dataset. **Next action:** request location of actual top/side MP4 files, ten-rater score XLSX, original cow ID and farm/date/session identifiers, dataset-specific rights and commercial/noncommercial terms.

## C. Independent pig play → veterinary outcome source — descriptive schema, but downloads return 403

- Source: [Dryad 10.5061/dryad.76hdr7t55](https://datadryad.org/dataset/doi:10.5061/dryad.76hdr7t55), source metadata **CC0**.
- Source README lists **nine litters** (five enrichment/play and four controls), 28 challenged pigs, plus independently measured post-viral-infection respiratory scores, skin lesions, body temperatures, viral loads and behavior measures. Clinical variables include ordinal respiratory severity and day counts, as distinct from play observed pre-/post-inoculation.
- This is a potentially strong **positive-welfare evidence** study, but we must not claim that PLY experimental treatment **is** the pig's positive emotion. Also a disease challenge with only nine litter experimental units makes predictive claims extremely uncertain.
- [Dryad site](https://datadryad.org/dataset/doi:10.5061/dryad.76hdr7t55) exposes the source workbook links, but direct download currently yields HTTP **403 Forbidden** for at least the behavior and clinical links. Our read-only source attempt [CI 38824216943](https://github.com/ob1-s/OpenFARM/actions/runs/38824216943) confirmed small XLSX file metadata but was not able to download the bytes. The file-level clinical outcome join is **not verified**; no new task approved. Seek legitimate author-supported access rather than treating the landing page as full download.
- **Status:** `METADATA_AND_DATA_DICTIONARY_VERIFIED / FILE_BYTES_ACCESS_PENDING`.

## Specific access/permissions questions to send

### Brazilian ten-rater cow dataset (Portuguese draft)

**Assunto:** Acesso a vídeos e planilha de dez avaliadores — claudicação de vacas leiteiras

Olá! Estou fazendo uma pesquisa aberta sobre modelos pequenos, rápidos e calibrados para análise de movimento e bem-estar animal. Encontrei o projeto de claudicação no Animal Welfare Science HUB, com 215 pares de vídeos e dez avaliações independentes por par.

Gostaria de saber se é possível acessar os MP4 (vista superior e lateral) e a planilha original com as notas de cada avaliador. O site descreve os arquivos, mas não consegui localizar o download. Seria especialmente útil saber se existem identificadores anonimizados de cada vaca, fazenda, sessão ou data, para separar treino e teste por animal/local e evitar vazamento.

Também poderiam confirmar a licença de uso dos vídeos e notas para (a) pesquisas e avaliações abertas sem fins comerciais e (b) eventual treinamento de modelos para uma aplicação comercial, sem redistribuir os vídeos? Podemos atribuir os autores e respeitar as condições informadas.

Obrigado!

### 2026 cow pose trajectories (English draft)

**Subject:** Research reuse terms and cow-level gait label provenance for your 272-trajectory dataset

Hello, I'm auditing public animal-welfare decision data for an open research effort on small, calibrated gait models. Thank you for publishing the 272 trajectories and matched locomotion scores. I verified 98 unique cow IDs, with several repeated clips per animal.

Would you clarify the permitted research training/evaluation and commercial machine-learning uses? The repository seems to have no LICENSE file. Also, are the underlying individual assessor ratings (rather than only `hard_vote`) and recording dates/site IDs available? Our evaluation would explicitly split by cow to avoid leakage and would attribute your data and study.

Thank you!

### Pig play / PRRSV experiment (English draft)

**Subject:** Dryad clinical/behavior Excel access and temporally joined animal outcomes

Hello, I'm researching independently validated animal welfare and positive behavior indicators. The Dryad dataset `10.5061/dryad.76hdr7t55` is exactly the type of source we aim to inspect: play behavior together with separately scored respiratory, lesion and immune outcomes.

The dataset page displays CC0 Excel files, but their file-stream download currently responds with HTTP 403 from our environment. Is there an alternative authorized way to access the original behavior and clinical XLSX files or a working download link? We will preserve pig, litter, treatment and time metadata separately and avoid interpreting play-treatment assignment as an emotion label.

Many thanks!

*These are drafts only. No emails have been sent.* 

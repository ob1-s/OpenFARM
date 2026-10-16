# OpenFARM — verified researcher contacts and access/rights requests

**Prepared 2026-10-15. These messages have NOT been sent.** No researcher has granted access, and no external repository has been modified. Contact details below are publicly listed by the corresponding research groups/papers. This is the handoff for obtaining data legally rather than inferring rights from public visibility.

## Priority 1 — Brazilian cow locomotion (10 raters, 215 top/side video pairs)

**Recipients:** `victoriapdgaia@gmail.com` (research-page contact for 3D source files) and/or `contact@animalwelfarehub.org` (platform contact).  
**Verified source:** https://animalwelfarehub.org/cows-project/ and https://animalwelfarehub.org/contact/  
**Status:** The 2D MP4/ratings repository is described but no real download link was visible in webpage HTML. We have **not** obtained files. The author's email on the project page specifically offers 3D files; do not assume it is the only correct contact for 2D.

**Subject:** Acesso ao conjunto de vídeos e às dez avaliações de claudicação de vacas — pesquisa OpenFARM

Olá,

Encontrei o projeto *Lameness in Dairy Cows* no Animal Welfare Science HUB. Estou conduzindo uma pesquisa aberta (OpenFARM) sobre modelos pequenos, eficientes e calibrados para interpretar observações de movimento e identificar possíveis indicadores de bem-estar animal.

O site descreve **215 observações com vídeos das vistas superior e lateral (430 MP4)** e avaliações de **dez especialistas por observação**, mas não consegui encontrar links funcionais para os MP4 ou a planilha de notas. Vocês poderiam indicar o local de download, ou uma forma autorizada de solicitar os arquivos?

Para evitar vazamento na avaliação, também seria importante saber se existem IDs anonimizados de **cada animal, fazenda, sessão e data**, e se a planilha inclui as **dez notas individuais**, não apenas a média, para avaliar desacordo entre especialistas.

Poderiam esclarecer a licença dos vídeos e das anotações para: (1) avaliações e treinamento em pesquisa aberta sem fins comerciais, sem republicar os vídeos; e (2) eventual treinamento ou uso comercial de modelos, com atribuição dos autores? Caso os usos tenham condições diferentes, podemos mantê-los separados.

Obrigado pela atenção e pelo trabalho de vocês!

## Priority 2 — Russello 2026 lameness gait trajectories

**Recipient:** `helena.russello@wur.nl` (given as Helena Russello's contact in the older official `hrussel/lameness-detection` README).  
**Verified source:** https://github.com/hrussel/lstm-lameness-detection (2026 dataset), https://github.com/hrussel/lameness-detection (2024 code), https://research.wur.nl/en/publications/lameness-detection-in-dairy-cows-using-pose-estimation-and-bidire/  
**Status:** 272 trajectories and score table already recovered, 98 cow IDs. The 2026 repo has no LICENSE. The 2024 repo has Apache 2.0, but that doesn't establish reuse terms for new trajectories.

**Subject:** Permission and evaluator-score provenance for your 272 cow gait trajectories

Hello,

I'm conducting an open research audit for **OpenFARM**, focused on small, fast, calibrated animal-welfare decision models. Thank you for publishing the 2026 pose trajectory and locomotion-score dataset.

We verified all **272** released sequences and corresponding scores, grouped into **98 identifiable cows**, many with repeated recordings. We'd like to use the source for carefully animal-heldout research training and evaluation, while preserving original attribution and avoiding republishing your source trajectories.

Could you clarify what **license and permitted uses apply specifically to the 2026 data**? The 2026 repository has no explicit LICENSE, although an earlier related code repository is Apache-2.0. We want to distinguish noncommercial research/benchmarking, sharing derived analysis/models, and possible future commercial training.

If available, may we also obtain the individual assessor votes behind `hard_vote`, the score rubric, session dates and farm identifiers? Those are needed to evaluate rater disagreement and avoid animal/session leakage.

Many thanks for your work and any guidance.

## Priority 3 — Dryad 2024 pig play and PRRSV clinical-outcome spreadsheets

**Recipients:** `karolina.steinerova@usask.ca` (corresponding author) and/or `yolande.seddon@usask.ca` (co-corresponding author).  
**Verified source:** https://pmc.ncbi.nlm.nih.gov/articles/PMC11443507/ and https://datadryad.org/dataset/doi%3A10.5061/dryad.76hdr7t55  
**Status:** Source is CC0 on Dryad; per-file links identified from metadata, but public downloadable bytes still inaccessible in our CI. Official dataset-wide download responds HTTP **401** (CI https://github.com/ob1-s/OpenFARM/actions/runs/38829231806). We **cannot** claim source access. We should also report a possible Dryad support issue rather than assuming author-imposed restriction.

**Subject:** Access to Dryad pig play and clinical data (10.5061/dryad.76hdr7t55)

Hello,

I'm researching behavioral indicators of positive animal welfare and their relationships with independent clinical outcomes for an open project, **OpenFARM**.

Your 2024 study on play opportunities and PRRSV resilience is unusually relevant because it provides behavior observations alongside respiratory signs, skin lesions and health outcomes, rather than assuming that play itself establishes emotional valence.

We're trying to inspect the original CC0 Dryad spreadsheets at https://doi.org/10.5061/dryad.76hdr7t55. The API metadata is publicly visible and lists the Excel files, but the official file download endpoints are inaccessible from our research environment, and the documented whole-dataset API download returns HTTP 401. Would you know of an authorized alternative download location or be able to ask Dryad to check access?

We would preserve the **pig, litter, treatment, and observation-time** identifiers so independent outcome tasks are not inflated by repeated observations or experimental-label shortcuts. We don't need any unpublished identifying information.

Thank you very much for making this valuable research available.

## Priority 4 — USP-linked sow locomotion (13 trained scorers, two farms)

**Recipients:** `tauanamariapaula@gmail.com` and/or `adroaldo.zanella@usp.br` (corresponding authors as printed in the original Scientific Reports paper); `contact@animalwelfarehub.org` if project routing is preferable.  
**Primary sources:** https://animalwelfarehub.org/sows-project/ ; https://doi.org/10.1038/s41598-024-62151-7  
**Status:** The project landing page mentions **two recording locations**, but the peer-reviewed paper describes this analyzed dataset as collected at **one commercial farm (TOPGEN)**. The paper reports **500 source sows; 364 lateral MP4s assessed by 13 experts; 11 dropped as unscorable; three raters excluded as outliers**. Dorsal labels were **inherited from the corresponding lateral video**, not independently scored. It states **data available from the corresponding author upon reasonable request**. The page's 2D dataset download is not accessible. Do **not** describe this as a proven cross-farm eval source until more independent material is supplied.

**Subject:** Dados do projeto de locomoção de matrizes suínas — vídeos, 13 avaliadores e identificação por granja

Olá,

Li o artigo de 2024 *Deep learning pose detection model for sow locomotion* e encontrei o repositório descrito no Animal Welfare Science HUB. Estou pesquisando modelos pequenos e calibrados para interpretar movimentos como indicadores de bem-estar animal (OpenFARM).

O artigo menciona 500 porcas inicialmente filmadas, 364 vídeos laterais avaliados por 13 especialistas, 11 gravações descartadas por dificuldades de pontuação, três avaliadores posteriormente excluídos e transferência da nota lateral para a vista dorsal correspondente. Seria possível acessar os MP4 pareados e **as notas originais de todos os avaliadores, incluindo discordâncias e exclusões**? Também seriam úteis IDs anonimizados de porca e sessão, bem como a regra exata de correspondência entre as duas vistas.

A página do projeto menciona registros na USP e na TOPGEN, mas o artigo descreve a amostra analisada na TOPGEN. **Existem amostras adicionais da USP, independentes do conjunto publicado, que poderiam permitir avaliação entre granjas?** Caso não existam, consideraremos a avaliação limitada a uma única granja.

Também poderiam informar a licença e condições para pesquisa aberta, eventual redistribuição apenas de métricas/partições, e eventual uso comercial de modelos treinados com esses dados? Podemos respeitar restrições distintas para cada uso.

Agradeço pela atenção!

## Evidence / outcome tracking

Do not assert any request was sent or answered until an actual communication tool or the user confirms it. If data arrive, record immutable file digests, source revision, media rights, label/rater protocol and independent groups in the [admission policy](DATA_APPROVAL_POLICY.md). No PR merge or training approval is automatic upon receiving a reply.

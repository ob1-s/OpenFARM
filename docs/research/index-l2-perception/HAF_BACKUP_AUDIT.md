# HAF Horse Pain/No-Pain - backup-lane source audit (Operator B, 2026-10-16)

Remote structural audit of the designated backup source (no archive download).
Result: **technically clean, but legally unusable as adopted data** (no license).

## Record facts (Zenodo 14541977, concept 14541976)

- Title: "HAF 2024 - Automatisk genkendelse af smerte hos heste - Horse pain/no pain - cropped images"
- Creators: Jensen, Dan Borge; Lindegaard, Casper; Gleerup, Karina; Schonberg Olsen, Lea (AU Viborg / HAF).
- Published 2024-12-22. **`rights: null` and no `license` field** - the record grants no usage rights.
- Single file: `Pain and NoPain.zip`, 233,786,527 B, md5 `be09a4ac6c1d3db3c4908cad1aeca075` (zenodo-installable `prime env install` not attempted pending rights decision).

## Machine-checked structure (byte-range central directory, verified decompressions)

- 606 members: **605 PNG (+1 `Thumbs.db` Windows junk file)** in `Pain and NoPain/`.
- Filenames: `cropped_{pain|nopain}_video{N}_1_{frame:02d}.png`
  - classes: **pain = 314, nopain = 291**
  - 22 video groups (naming allows multiple horses per video, but the record
    description confirms **11 horses over 22 videos**; no horse-ID column exists,
    so video-disjoint is the only available grouping proxy for horse-disjoint).
- Sampled images: ~400-712 x 720 px, 128-623 KB, genuine photographs (decompress
  + CRC32 + PNG IHDR verified on 5 members).

## Positive

- Pre-cropped centered faces, binary pain/no-pain, expert-judged (B. Jensen HAF),
  human faces filtered out - mechanism directly matches an Owner-approved task
  (pain-associated cue).
- Frame pointers recreate the range: video groups are the natural across-animal split.

## Blockers

1. No license on the record - the data cannot be redistributed or made the base of
   a derived corpus without author permission. Per DATA_APPROVAL_POLICY this is
   adopt-blocked; gather-only.
2. Frame-pointer coupling to horse identity is not verifiable (11 horses / 22
   videos claimed, no explicit map) - even with permission, an animal-disjoint
   split would need author confirmation of the video-horse adjacency.
3. `Thumbs.db` inside the data zip (junk; exclude on ingest).

## Verdict

Keep GrimACE as the adopted L2 perception benchmark. HAF stays the pre-vetted
backup: if the authors grant a license, `Pain and NoPain.zip` (md5 above) is fully
structurally validated by this audit and a video-disjoint split is deployable in
roughly the same shape as the GrimACE pipeline (lib is source-agnostic).

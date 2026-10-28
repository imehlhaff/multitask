# Notes on CAP

**[Mon Oct 19, 2026]: downloaded party manifesto/platform datasets**

Source: https://www.comparativeagendas.net/datasets_codebooks (files hosted at `https://minio.la.utexas.edu/compagendas/datasetfiles/`)

Chose the English sentence-level party documents, since they are the closest match to CMP (English manifesto quasi-sentences, median ~140 chars). `00_download.py` saves them to `raw/`.

| File | Country | Rows | Median chars | Years | Text column |
|---|---|---|---|---|---|
| `uk_manifestos_v1__1.csv` | UK | 36,293 | 75 | 1983–2015 | `description` |
| `irl_manifestos_data_1981_2020_utf8.csv` | Ireland | 66,292 | 107 | 1981–2020 | `text` |
| `Dem_partyplatform_20.2.csv` | US (Democratic) | 19,694 | 131 | 1948–2024 | `description` |
| `US-Party-Rep_Platform_19.3_0.csv` | US (Republican) | 20,156 | 125 | 1948–2024 | `description` |

Labels: `majortopic` (21 CAP major topics) and `subtopic` (~220), one per row.

Master codebook: https://www.comparativeagendas.net/pages/master-codebook 

## Complementarities with CMP
**CMP**: A comprehensive collection of political party election manifestos from more than 50 countries covering democratic elections since 1945. Originally started as the Comparative Manifestos Project (CMP) and continued via MARPOR, it uses quantitative content analysis to measure party policy positions and issue attention across 56 categories.

**CAP**: An international collaborative research initiative that tracks, measures, and compares government policy-making and agenda-setting activities across different countries and time periods.
- **Same kind of text, different labels.** Both are party documents coded sentence by sentence. CMP codes position within a policy area (e.g. `504` welfare expansion vs `505` welfare limitation), while CAP codes only the topic. 
- **Taxonomies roughly line up.** Many CMP codes map approximately onto CAP major topics, e.g. `501` environmental protection → 7 Environment, `506`/`507` education → 6 Education, `605` law and order → 12 Law & Crime, `703` agriculture → 4 Agriculture, `104`/`105` military → 16 Defense. The mapping is many-to-many and incomplete: CMP's 2xx/3xx codes (democracy, political system) and 6xx codes (national way of life, morality) have no clean CAP equivalent. It could be used for consistency checks or error analysis, but not as a substitute label.
- **Different label structures.** CMP is flat multi-label (75 binary indicators). CAP is single-label and hierarchical (major → subtopic).
- **Wider coverage.** CMP's training data looks like UK, Australian and NZ manifestos. CAP adds the US and Ireland and goes back to 1948, so the encoder sees more countries and periods.
- **More data.** About 142k CAP sentences vs 52k CMP train rows. Whether this helps CMP's rare codes through the shared encoder is an open question to test.
- **Doubly coded sentences.** UK sentences that appear in both datasets would carry a CMP code and a CAP topic. They are useful for checking the mapping above, but they are also the main leakage risk.

## Codebooks (`codebooks/`)
- `UK_manifestos_codebook_v1.1.pdf`: variables plus Froio's coding handbook (1983–2010) and a note on the 2015 coding. The quasi-sentence definition is taken directly from the CMP/MARPOR handbook, so the coding unit is the same as CMP. CAP coders are told to split when in doubt, which may explain the shorter UK sentences (median 75 vs ~140 chars in CMP). Reliability for 2015: Krippendorff's alpha 0.85 (sentence splitting), 0.90 (major topic), 0.70 (subtopic). The handbook lists UK-specific topic names (e.g. 6 "Education and Culture", no 23), but the data uses the master codebook, which has 23 Cultural Policy.
- `IPAP CAP codebook v1.2.xlsx`: Irish major topics and subtopics. Irish-specific subtopics (e.g. 1922 Northern Ireland, 2301 Irish Language) have a column mapping them back to master codebook subtopics. Use it to harmonize subtopics with the other files. Also documents the EU, Covid-19 and Northern Ireland dummies.
- `IPAP_mainfestos_variables_for_CAP.pdf`: one-page note. The full variable descriptions are in a data report on irishpolicyagendas.eu (not downloaded).
- `PartyPlatformsCodebookJuly2016.pdf` (`_1` is an identical duplicate): one CAP code per quasi-sentence, split at periods and semicolons. Says 1948–2008, but the data now runs to 2024. No newer version is linked on the CAP site.
  - `pap_*` columns aren't covered here, but other US codebooks from the same team (`CRS_Codebook_19.2.pdf`, `CQ_Almanac_Codebook.pdf` at `https://minio.la.utexas.edu/compagendas/codebookfiles/`) define them: `pap_majortopic`/`pap_subtopic` are codes in the US Policy Agendas Project scheme, and `majortopic`/`subtopic` are the same codes converted to the CAP master scheme. US topic definitions: https://www.comparativeagendas.net/us
  - `-555` isn't defined in any CAP codebook checked. Possibly a missing/uncoded value, unconfirmed. Contact: policyagendas@gmail.com
- None of these include the CMP/MARPOR codebook, so the CMP → CAP mapping above is still unverified.

## Things to handle in cleaning
- The "no policy content" code differs: `0` in UK and US, `99` in Irish. Harmonize before pooling.
- US platforms: use `majortopic` (CAP master), not `pap_majortopic` (US Policy Agendas). They differ on ~130 Dem / ~1,090 Rep rows: PAP 5 → CAP 13, PAP 6 → CAP 23, and PAP `-555` → CAP `0`. `majortopic` still has `-555` (59 Dem / 131 Rep rows), blanks (41 Dem rows) and a few codes outside the scheme (24, 26, 27, 28, 31; ≤11 rows per file). Drop these.
- The Irish file has mojibake in places (e.g. `TaoiseachÂ’s`).
- UK (and maybe Irish) manifestos may overlap with sentences in CMP. Split by document to avoid leakage across datasets.

## Not used
- Non-English datasets, including the manifestos from other countries
- Bills, laws, hearings and headlines (short titles/descriptions)
- Budgets and most-important-problem surveys (no text)
- Australian files (malformed CSVs)
- UK speech from the throne and US State of the Union: sentence-level, but government speeches rather than party documents. Possible later additions.


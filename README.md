# Eesti–inglise koodivahetuse korpus ja keeletuvastajate võrdlus

Lõputöö reprodutseerimiseks vajalik: kaks skripti — üks
Redditi korpuse kogumiseks, teine kõikide 12 keeletuvastaja
hindamiseks Redditi ja vestluskorpusel.

---

## Paigaldus

Nõuab **Python ≥ 3.11**.

```bash
python -m venv .venv
source .venv/bin/activate                     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                              # teeb src/ moodulid imporditavaks
python scripts/download_models.py             # lid.176.bin (~125 MB) + HF mudelid (~500 MB)
python evaluation/environment_check.py        # kontrolli, et kõik 12 tuvastajat on imporditavad
```

---

## Reprodutseerimine

### 1. Hinda kõiki keeletuvastajaid mõlemas korpuses

```bash
python evaluation/full_identifier_report.py
```

Skript kasutab 12. tuvastajat (`pycld2`, `heliport`, `lingua`,
`langid`, `fasttext`, `fast-langdetect`, `nb-nordic-lid`, `glotlid`,
`masklid`, `lingua-mixed`, `masklid-cs`, `ane-lid`) 

**Sisendid** (vaikimisi):
- `data/chat/chat_tsv/` — 110 annoteeritud vestlusfaili
- `evaluation/reddit_margendus.xlsx` — Redditi kuldstandard

**Väljundid**:
- `evaluation/results/chat_tsv_full_report.xlsx`
- `evaluation/results/reddit_full_report.xlsx`

Mõlemas Exceli failis on kaks lehte: `Summary` (täpsus / saagis / F1
iga tuvastaja kohta) ja `SentenceView` (lause kohta kuldstandardi ja
ennustuste võrdlus). 

Tööaeg esimesel käivitamisel ≈ 5–15 min 

### 2. Värske Redditi korpuse kogumine (valikuline)

```bash
python scripts/run_multi_identifier.py \
    --limit 2500 \
    --output data/reddit/eesti_reddit_codeswitch_uus.xlsx \
    --min-eng-words 2 \
    --resume \
    --require-pretrained-confirmation \
    --pretrained-confirm-tools ane-lid,lingua-mixed \
    --verbose
```

**Väljund**: kolme lehega Excel — `Sentences`, `IdentifierScores`,
`RunStats`.

`--resume` kirjutab `<output>.checkpoint.json` faili, et kogumist saaks
katkestamise järel jätkata. 

---

## Kataloogi struktuur

```
.
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
│
├── src/                              ← põhiteek (imporditav pip install -e . kaudu)
│   ├── config.py                     ← otsingusõnad, alamfoorumid
│   ├── scraper.py                    ← Redditi avaliku JSON API klient
│   ├── sentence_processor.py         ← teksti puhastus + lausete poolitamine
│   ├── language_detector.py          
│   ├── identifier_suite.py           ← 12 tuvastaja ümbrised
│   ├── exporter.py                   ← Exceli väljundi loogika
│   └── utils.py                      
│
├── scripts/
│   ├── run_multi_identifier.py       ← Redditist lausete kogumine
│   └── download_models.py            ← mudelite tõmbamine
│
├── evaluation/
│   ├── full_identifier_report.py     ← mõlema korpuse hindamine
│   ├── tsv_identifier_evaluator.py   
│   ├── environment_check.py          
│   ├── reddit_margendus.xlsx         ← Redditi kuldstandard
│   └── MaskLID/                      ← iteratiivne maskeerimise tuvastaja (ACL 2024)
│       ├── masklid.py
│       ├── README.md
│       └── LICENSE
│
├── data/
│   ├── README.md                     ← andmete kirjeldus ja kasutustingimused
│   ├── reddit/
│   │   └── eesti_reddit_codeswitch_2500.xlsx     ← lõputöös kasutatud Redditi korpus
│   └── chat/chat_tsv/                            ← 110 märgendatud vestlusfaili
│
└── expected_results/
    ├── reddit_full_report.xlsx                   ← lõputöös kajastatud Redditi mõõdikud
    └── chat_tsv_full_report.xlsx                 ← lõputöös kajastatud vestluskorpuse mõõdikud
```

---

## Andmed

Mõlema korpuse kirjeldus on failis [data/README.md](data/README.md).

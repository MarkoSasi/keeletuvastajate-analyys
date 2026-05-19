# Andmed

Selles kataloogis on kaks lõputöös kasutatud korpust. Lähtekoodi
litsents (kui see on lisatud ülemise tasandi `LICENSE` failina) kehtib
ainult koodile — siinsetele andmetele kohaldatakse allpool kirjeldatud
tingimusi.

```
data/
├── reddit/
│   └── eesti_reddit_codeswitch_2500.xlsx
└── chat/
    └── chat_tsv/                       # 110 annoteeritud TSV-faili
```

## `data/reddit/` — r/Eesti koodivahetuse korpus

**Allikas.** Postitused ja kommentaarid, mis on skreibitud
[r/Eesti](https://www.reddit.com/r/Eesti/)-st Redditi avalike,
autentimata JSON-otspunktide kaudu (`/r/Eesti/<category>.json` ja
`/<permalink>.json`). Skreiperi kood asub failis `src/scraper.py`;
kogumise loogika ja koodivahetuse filtreerimine on failis
`scripts/run_multi_identifier.py`.

**Filtreerimiskonveier.** Iga lausekandidaat läbib range
koodivahetuse värava (`src/language_detector.py:is_codeswitched`),
mis nõuab ≥1 eesti ankurit, ≥1 inglise funktsioonisõna, kokku ≥2
inglise sõna ning välistab brändi- ja laensõnad. Lisaks nõutakse, et
vähemalt üks eelõpetatud tuvastaja (näiteks `ane-lid`, `lingua-mixed`)
kinnitab inglise lõigu.

**Fail.** `eesti_reddit_codeswitch_2500.xlsx` — kolme lehega Excel:
- `Sentences` — üks rida lause kohta (`id`, `sentence`, `source_url`,
  `source_type`, `word_count`, `anchor_cs_parts`, `consensus_main`,
  ning iga tuvastaja veerud: `<tool>_main`, `<tool>_cs_parts`,
  `<tool>_token_f1_vs_anchor`).
- `IdentifierScores` — iga tuvastaja koondmõõdikud.
- `RunStats` — kogumise metaandmed (kuupäev, parameetrid, allikate
  jaotus).

**`source_url` poliitika.** Permalinkid säilitatakse muutmata kujul,
et iga lause päritolu oleks kontrollitav. Mistahes lingi avamine viib
avaliku Redditi kommentaari juurde ja paljastab selle autori.
Sellest korpusest tuletatud töid avaldavad kasutajad peaksid tutvuma
Redditi
[Public Content Policy](https://redditinc.com/policies/public-content-policy)-ga
ja kaaluma, kas need URL-id oma tuletatud andmestikus säilitada,
räsida või eemaldada.

**Taaskasutamise märkus.** Avalike postituste lausetasandi
tekstilõigud on lisatud mitteäriliseks akadeemiliseks korratavuseks.
Tekst kuulub algsetele Redditi kasutajatele; sellest lõputöö ulatusest
väljapoole ulatuvaks edasilevitamiseks tuleb tutvuda Redditi
tingimustega.

## `data/chat/chat_tsv/` — eesti–inglise vestluse koodivahetuse korpus

**Allikas.** Eesti noorte (eri vanuse- ja sookategooriatest) Messengeri
ja Discord'i vestluste anonümiseeritud TSV-eksport. 110 faili.
Annoteerimist (siltide `<ann_lang='LANG'>span</>` lisamist) ei tehtud
selle töö raames.

**Subjektide anonümiseerimine.** Failinimedes on koodid nagu
`Ant10m_428309_Messenger_1`. Esimesed kolm tähte on omavalitsuste
koodid (Ant = Antsla, Krs = Karksi, Trt = Tartu, Tln = Tallinn jne);
numbrid kodeerivad klassi ja sugu; lõpus olev täisarv on
anonümiseeritud osaleja ID. Päris nimesid andmetes ei esine.

**TSV-vorming.** Üks fail vestlussessiooni kohta. Veerud:

| Veerg               | Tähendus                                                     |
|---------------------|--------------------------------------------------------------|
| `Jutt`              | sessiooni märgend (`jah` põhivestluse jaoks)                 |
| `Aeg`               | ajatempel (eesti kuupäevavorming)                            |
| `Osaleja`           | osaleja ID (numbriline)                                      |
| `Puhas_tekst`       | annotatsioonita sõnumitekst                                  |
| `Margendatud_tekst` | sõnumi tekst koos sisesete keelesiltidega                    |

Hindaja `evaluation/full_identifier_report.py` loeb ainult ridu, kus
`Jutt == "jah"`, ja parsib `Margendatud_tekst` veerust kuldstandardi
tokenite kaupa.

**Sisesete siltide vorming.** Võõrkeelne lõik on ümbritsetud sildiga
`<ann_lang='LANG'>span</>`, kus `LANG` on `eng` inglise või `est`
eesti keele jaoks. Sildita tekst loetakse selle sessiooni
maatrikskeeleks. Näide:

```
You teed <ann_lang='eng'>my day</>
jou jah  → <ann_lang='eng'>jou</> jah
```

## Kasutus

Mõlemat korpust kasutab vaikimisi sama skript:

```bash
python evaluation/full_identifier_report.py
```

Sisendteed saab üle kirjutada `--tsv-dir` ja `--reddit-xlsx` lipudega
(vt `--help`). Väljundid kirjutatakse vaikimisi `evaluation/results/`
kataloogi.

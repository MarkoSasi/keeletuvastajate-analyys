"""
Täielik keeletuvastajate hindamine eesti-inglise koodivahetuse jaoks.

Käivitab kõik saadaolevad eelnevalt treenitud keele tuvastajad kahel andmeallikal:
  1. chat_tsv/  – märgendatud TSV-failid (filtreeritud Jutt == "jah" järgi)
  2. evaluation/reddit_margendus.xlsx – märgendatud Redditi laused

Loob Exceli-aruanded järgmiste lehtedega:
  1. leht (SentenceView): lausepõhised kuld- + ennustatud inglise lõigud
  2. leht (Summary):      iga tuvastaja mõõdikute tabel
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Teede seadistus
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.tsv_identifier_evaluator import (  # noqa: E402
    _safe_text,
    parse_annotated_tokens,
    label_spans,
    overlap_span_tp,
)


# ---------------------------------------------------------------------------
# Andmestruktuurid
# ---------------------------------------------------------------------------
@dataclass
class SentenceData:
    """Üksik lause koos kuldamärgendusega."""
    id: int
    file: str
    sentence: str
    tokens: List[str]
    gold_labels: List[str]
    gold_spans: List[Tuple[str, int, int]]
    gold_has_eng: bool


@dataclass
class ToolMetrics:
    """Üksiku tuvastaja mõõdikud."""
    token_total: int = 0
    token_correct: int = 0
    est_total: int = 0
    est_correct: int = 0
    eng_tp: int = 0
    eng_fp: int = 0
    eng_fn: int = 0
    cs_sentences_gold: int = 0
    cs_sentences_detected: int = 0
    span_overlap_tp: int = 0
    span_overlap_pred: int = 0
    span_overlap_gold: int = 0


def spans_to_text(spans: List[Tuple[str, int, int]], tokens: List[str]) -> str:
    """Renderda ENG-lõigud loetavateks tekstifragmentideks."""
    if not spans:
        return "-"
    parts = []
    for label, start, end in spans:
        frag = " ".join(tokens[start:end]).strip()
        if frag:
            parts.append(frag)
    return " | ".join(parts) if parts else "-"


# ---------------------------------------------------------------------------
# Mõõdikute abifunktsioonid
# ---------------------------------------------------------------------------
def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


# ---------------------------------------------------------------------------
# Tuvastajate registreerimine (kõik 12 eeltreenitud mudelit)
# ---------------------------------------------------------------------------
def register_detectors() -> Tuple[
    Dict[str, Callable[[str], str]],
    Dict[str, Callable[[List[str]], List[str]]],
]:
    """Registreeri kõik saadaolevad eelnevalt treenitud keele tuvastajad."""
    tools_token: Dict[str, Callable[[str], str]] = {}
    tools_sequence: Dict[str, Callable[[List[str]], List[str]]] = {}

    # ---- Abifunktsioon märki -> sõne-märgendi vastendamiseks ----
    def _labels_from_char_spans(
        tokens: List[str], spans: List[Tuple[int, int]]
    ) -> List[str]:
        labels = ["EST"] * len(tokens)
        if not tokens or not spans:
            return labels
        offsets: List[Tuple[int, int]] = []
        pos = 0
        for tok in tokens:
            start = pos
            end = start + len(tok)
            offsets.append((start, end))
            pos = end + 1
        for i, (tok_s, tok_e) in enumerate(offsets):
            for sp_s, sp_e in spans:
                if sp_e > tok_s and sp_s < tok_e:
                    labels[i] = "ENG"
                    break
        return labels

    def _find_phrase_spans(
        text: str, phrases: List[str]
    ) -> List[Tuple[int, int]]:
        text_low = text.lower()
        spans: List[Tuple[int, int]] = []
        for phrase in phrases:
            phrase_low = phrase.strip().lower()
            if not phrase_low:
                continue
            start = 0
            while True:
                idx = text_low.find(phrase_low, start)
                if idx == -1:
                    break
                end = idx + len(phrase_low)
                left_ok = idx == 0 or not text_low[idx - 1].isalnum()
                right_ok = end == len(text_low) or not text_low[end].isalnum()
                if left_ok and right_ok:
                    spans.append((idx, end))
                start = idx + 1
        return spans

    # ---- pycld2 ----
    try:
        import pycld2
        def tok_pycld2(word: str) -> str:
            if len(word) < 2:
                return "EST"
            try:
                _, _, details = pycld2.detect(word, bestEffort=True)
                lang_code = details[0][1]
                return "ENG" if lang_code == "en" else "EST"
            except Exception:
                return "EST"
        tools_token["pycld2"] = tok_pycld2
    except Exception as e:
        print(f"  ⚠ pycld2 not available: {e}")

    # ---- HeLI-OTS (heliport) ----
    try:
        import heliport
        heli_detector = heliport.Identifier()
        def tok_heliport(word: str) -> str:
            if len(word) < 2:
                return "EST"
            try:
                res = heli_detector.identify(word)
                return "ENG" if res == "eng" else "EST"
            except Exception:
                return "EST"
        tools_token["heliport"] = tok_heliport
    except Exception as e:
        print(f"  ⚠ heliport not available: {e}")

    # ---- nb-nordic-lid ----
    try:
        import fasttext
        from huggingface_hub import hf_hub_download
        nb_path = hf_hub_download(
            repo_id="NbAiLab/nb-nordic-lid", filename="nb-nordic-lid.ftz"
        )
        fasttext.FastText.eprint = lambda *a, **k: None
        nb_model = fasttext.load_model(nb_path)

        def tok_nb_nordic_lid(word: str) -> str:
            if len(word) < 2:
                return "EST"
            try:
                pred = (
                    nb_model.predict(word, k=1)[0][0]
                    .replace("__label__", "")
                    .lower()
                )
                return "ENG" if pred in ("en", "eng", "english") else "EST"
            except Exception:
                return "EST"
        tools_token["nb-nordic-lid"] = tok_nb_nordic_lid
    except Exception as e:
        print(f"  ⚠ nb-nordic-lid not available: {e}")

    # ---- GlotLID ----
    try:
        import fasttext
        from huggingface_hub import hf_hub_download
        glot_path = hf_hub_download(
            repo_id="cis-lmu/glotlid", filename="model.bin"
        )
        fasttext.FastText.eprint = lambda *a, **k: None
        glot_model = fasttext.load_model(glot_path)

        def tok_glotlid(word: str) -> str:
            if len(word) < 2:
                return "EST"
            try:
                pred = (
                    glot_model.predict(word, k=1)[0][0]
                    .replace("__label__", "")
                    .lower()
                )
                return "ENG" if "eng" in pred else "EST"
            except Exception:
                return "EST"
        tools_token["glotlid"] = tok_glotlid
    except Exception as e:
        print(f"  ⚠ glotlid not available: {e}")

    # ---- Lingua (token + sequence) ----
    try:
        from lingua import Language, LanguageDetectorBuilder

        ling = LanguageDetectorBuilder.from_languages(
            Language.ESTONIAN, Language.ENGLISH
        ).build()

        def tok_lingua(word: str) -> str:
            if len(word) < 2:
                return "EST"
            lang = ling.detect_language_of(word)
            return "ENG" if lang == Language.ENGLISH else "EST"

        tools_token["lingua"] = tok_lingua

        def seq_lingua_mixed(tokens: List[str]) -> List[str]:
            sentence = " ".join(tokens)
            try:
                parts = ling.detect_multiple_languages_of(sentence) or []
                eng_spans: List[Tuple[int, int]] = []
                for part in parts:
                    lang_obj = getattr(part, "language", None)
                    lang_name = str(getattr(lang_obj, "name", "")).upper()
                    if lang_name != "ENGLISH":
                        continue
                    s = getattr(part, "start_index", None)
                    e = getattr(part, "end_index", None)
                    if s is None or e is None:
                        continue
                    s_i, e_i = int(s), int(e)
                    if e_i > s_i:
                        eng_spans.append((s_i, e_i))
                if eng_spans:
                    return _labels_from_char_spans(tokens, eng_spans)
            except Exception:
                pass
            return [tok_lingua(tok) for tok in tokens]

        tools_sequence["lingua-mixed"] = seq_lingua_mixed
    except Exception as e:
        print(f"  ⚠ lingua not available: {e}")

    # ---- langid ----
    try:
        import langid as _langid
        _langid.set_languages(["et", "en"])

        def tok_langid(word: str) -> str:
            if len(word) < 2:
                return "EST"
            lang, _ = _langid.classify(word)
            return "ENG" if lang == "en" else "EST"

        tools_token["langid"] = tok_langid
    except Exception as e:
        print(f"  ⚠ langid not available: {e}")

    # ---- fasttext ----
    try:
        import fasttext
        ft_path = ROOT_DIR / "lid.176.bin"
        if ft_path.exists():
            fasttext.FastText.eprint = lambda *a, **k: None
            ft_model = fasttext.load_model(str(ft_path))

            def tok_fasttext(word: str) -> str:
                if len(word) < 2:
                    return "EST"
                pred = (
                    ft_model.predict(word, k=1)[0][0].replace("__label__", "")
                )
                return "ENG" if pred == "en" else "EST"

            tools_token["fasttext"] = tok_fasttext
    except Exception as e:
        print(f"  ⚠ fasttext not available: {e}")

    # ---- fast-langdetect ----
    try:
        from fast_langdetect import detect as fld_detect

        def tok_fastlangdetect(word: str) -> str:
            if len(word) < 2:
                return "EST"
            res = fld_detect(word)
            if res and len(res) > 0:
                lang = str(res[0].get("lang", "et"))
                if lang in ("en", "eng"):
                    return "ENG"
            return "EST"

        tools_token["fast-langdetect"] = tok_fastlangdetect
    except Exception as e:
        print(f"  ⚠ fast-langdetect not available: {e}")

    # ---- MaskLID ----
    try:
        masklid_dir = ROOT_DIR / "evaluation" / "MaskLID"
        if str(masklid_dir) not in sys.path:
            sys.path.append(str(masklid_dir))
        from masklid import MaskLID

        masklid_model_path = ROOT_DIR / "lid.176.bin"
        if masklid_model_path.exists():
            masklid = MaskLID(
                str(masklid_model_path),
                languages=["__label__et", "__label__en"],
            )

            def tok_masklid(word: str) -> str:
                if len(word) < 2:
                    return "EST"
                try:
                    lab, _ = masklid.predict(word, k=1)
                    if lab:
                        top = str(lab[0]).replace("__label__", "")
                        return "ENG" if top == "en" else "EST"
                except Exception:
                    pass
                return "EST"

            def seq_masklid_cs(tokens: List[str]) -> List[str]:
                sentence = " ".join(tokens)
                try:
                    cs = masklid.predict_codeswitch(
                        sentence, beta=10, alpha=5, min_prob=0.3, min_length=1
                    )
                except Exception:
                    return [tok_masklid(tok) for tok in tokens]
                eng_text = ""
                if isinstance(cs, dict):
                    eng_text = str(
                        cs.get("__label__en", "") or cs.get("en", "")
                    ).strip()
                if not eng_text:
                    return [tok_masklid(tok) for tok in tokens]
                candidate_phrases = [
                    p.strip()
                    for p in re.split(r"[;|,]+", eng_text)
                    if p.strip()
                ]
                if not candidate_phrases:
                    candidate_phrases = [eng_text]
                spans = _find_phrase_spans(sentence, candidate_phrases)
                if not spans:
                    spans = _find_phrase_spans(sentence, [eng_text])
                if not spans:
                    return [tok_masklid(tok) for tok in tokens]
                return _labels_from_char_spans(tokens, spans)

            tools_token["masklid"] = tok_masklid
            tools_sequence["masklid-cs"] = seq_masklid_cs
    except Exception as e:
        print(f"  ⚠ masklid not available: {e}")

    # ---- AnE-LID ----
    try:
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
            pipeline,
        )

        model_id = "igorsterner/AnE-LID"
        ane_tokenizer = AutoTokenizer.from_pretrained(model_id)
        ane_model = AutoModelForTokenClassification.from_pretrained(model_id)
        ane = pipeline(
            "token-classification",
            model=ane_model,
            tokenizer=ane_tokenizer,
            aggregation_strategy="simple",
        )

        def seq_ane_lid(tokens: List[str]) -> List[str]:
            sentence = " ".join(tokens)
            try:
                out = ane(sentence)
            except Exception:
                return ["EST"] * len(tokens)
            spans: List[Tuple[int, int]] = []
            for ent in out:
                group = str(
                    ent.get("entity_group", ent.get("entity", ""))
                ).lower()
                if "english" in group and "not" not in group:
                    s = int(ent.get("start", -1))
                    e = int(ent.get("end", -1))
                    if e > s >= 0:
                        spans.append((s, e))
            return _labels_from_char_spans(tokens, spans)

        tools_sequence["ane-lid"] = seq_ane_lid
    except Exception as e:
        print(f"  ⚠ ane-lid not available: {e}")

    return tools_token, tools_sequence


# ---------------------------------------------------------------------------
# Andmete laadimine
# ---------------------------------------------------------------------------
def load_chat_tsv(tsv_dir: Path, min_words: int = 1) -> List[SentenceData]:
    """Laadi märgendatud laused vestluse TSV-failidest."""
    files = sorted(tsv_dir.glob("*.tsv"))
    if not files:
        raise FileNotFoundError(f"No TSV files in {tsv_dir}")

    sentences: List[SentenceData] = []
    skipped = 0
    for file in files:
        df = pd.read_csv(
            file, sep="\t", dtype=str, keep_default_na=False, encoding="utf-8"
        )
        for _, row in df.iterrows():
            jutt = str(row.get("Jutt", "")).strip().lower()
            if jutt != "jah":
                continue
            annotated = _safe_text(row.get("Margendatud_tekst", ""))
            if not annotated:
                continue
            tokens, gold_labels = parse_annotated_tokens(annotated)
            if not tokens or len(tokens) < min_words:
                skipped += 1
                continue
            gold_spans = label_spans(gold_labels)
            gold_has_eng = any(lbl == "ENG" for lbl in gold_labels)
            sentences.append(SentenceData(
                id=len(sentences) + 1,
                file=file.name,
                sentence=" ".join(tokens),
                tokens=tokens,
                gold_labels=gold_labels,
                gold_spans=gold_spans,
                gold_has_eng=gold_has_eng,
            ))
    print(f"  chat_tsv: loaded {len(sentences)} sentences (skipped {skipped} short)")
    return sentences


def load_reddit_xlsx(xlsx_path: Path, min_words: int = 1) -> List[SentenceData]:
    """Laadi märgendatud laused failist reddit_margendus.xlsx."""
    if not xlsx_path.exists():
        raise FileNotFoundError(f"File not found: {xlsx_path}")

    df = pd.read_excel(xlsx_path, engine="openpyxl", dtype=str)
    df = df.fillna("")

    sentences: List[SentenceData] = []
    skipped = 0
    for idx, row in df.iterrows():
        annotated = _safe_text(row.get("Margendatud_tekst", ""))
        if not annotated:
            continue
        tokens, gold_labels = parse_annotated_tokens(annotated)
        if not tokens or len(tokens) < min_words:
            skipped += 1
            continue
        gold_spans = label_spans(gold_labels)
        gold_has_eng = any(lbl == "ENG" for lbl in gold_labels)
        sentences.append(SentenceData(
            id=len(sentences) + 1,
            file=xlsx_path.name,
            sentence=" ".join(tokens),
            tokens=tokens,
            gold_labels=gold_labels,
            gold_spans=gold_spans,
            gold_has_eng=gold_has_eng,
        ))
    print(f"  reddit:   loaded {len(sentences)} sentences (skipped {skipped} short)")
    return sentences


# ---------------------------------------------------------------------------
# Põhihinnang
# ---------------------------------------------------------------------------
def evaluate_sentences(
    sentences: List[SentenceData],
    tools_token: Dict[str, Callable],
    tools_sequence: Dict[str, Callable],
    source_label: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Käivita kõik tuvastajad lausetel ja koosta DataFrame-id."""

    # Järjestus: kõigepealt jada-tööriistad, seejärel sõne-tööriistad
    all_tools = {**tools_sequence, **tools_token}
    tool_names = list(all_tools.keys())

    metrics = {name: ToolMetrics() for name in tool_names}
    sentence_rows: List[Dict[str, object]] = []

    total = len(sentences)
    t0 = time.time()

    for si, sd in enumerate(sentences, 1):
        row: Dict[str, object] = {
            "id": sd.id,
            "file": sd.file,
            "sentence": sd.sentence,
            "gold_cs_parts": spans_to_text(sd.gold_spans, sd.tokens),
        }

        for tool_name in tool_names:
            m = metrics[tool_name]

            # Ennustused
            if tool_name in tools_sequence:
                pred_labels = tools_sequence[tool_name](sd.tokens)
            else:
                pred_labels = [tools_token[tool_name](tok) for tok in sd.tokens]

            if len(pred_labels) != len(sd.gold_labels):
                row[f"{tool_name}_cs_parts"] = "MISMATCH"
                continue

            pred_spans = label_spans(pred_labels)
            pred_has_eng = any(lbl == "ENG" for lbl in pred_labels)

            # --- Mõõdikud ---
            m.token_total += len(sd.gold_labels)
            for g, p in zip(sd.gold_labels, pred_labels):
                if g == p:
                    m.token_correct += 1
                if g == "EST":
                    m.est_total += 1
                    if p == "EST":
                        m.est_correct += 1
                if p == "ENG" and g == "ENG":
                    m.eng_tp += 1
                elif p == "ENG" and g != "ENG":
                    m.eng_fp += 1
                elif p != "ENG" and g == "ENG":
                    m.eng_fn += 1

            if sd.gold_has_eng:
                m.cs_sentences_gold += 1
                if pred_has_eng:
                    m.cs_sentences_detected += 1

            # Lõikude kattuvus
            m.span_overlap_gold += len(sd.gold_spans)
            m.span_overlap_pred += len(pred_spans)
            m.span_overlap_tp += overlap_span_tp(sd.gold_spans, pred_spans)

            # --- Lausepõhine väljund ---
            row[f"{tool_name}_cs_parts"] = spans_to_text(pred_spans, sd.tokens)

        sentence_rows.append(row)

        if si % 200 == 0 or si == total:
            elapsed = time.time() - t0
            rate = si / elapsed if elapsed > 0 else 0
            print(f"  [{source_label}] {si}/{total} sentences ({rate:.0f}/s)")

    # --- Koosta kokkuvõte ---
    summary_rows = []
    for tool_name in tool_names:
        m = metrics[tool_name]
        if m.token_total == 0:
            continue

        token_acc = 100.0 * m.token_correct / m.token_total
        est_acc = 100.0 * m.est_correct / m.est_total if m.est_total else 0.0
        eng_p, eng_r, eng_f1 = _prf(m.eng_tp, m.eng_fp, m.eng_fn)
        cs_recall = (
            100.0 * m.cs_sentences_detected / m.cs_sentences_gold
            if m.cs_sentences_gold
            else 0.0
        )

        # Lõikude kattuvuse F1 liitskoori jaoks
        ov_p = m.span_overlap_tp / m.span_overlap_pred if m.span_overlap_pred else 0.0
        ov_r = m.span_overlap_tp / m.span_overlap_gold if m.span_overlap_gold else 0.0
        ov_f1 = 2 * ov_p * ov_r / (ov_p + ov_r) if (ov_p + ov_r) else 0.0

        composite = (
            0.30 * token_acc
            + 0.25 * est_acc
            + 0.45 * (ov_f1 * 100.0)
        )

        summary_rows.append({
            "identifier": tool_name,
            "token_accuracy_pct": round(token_acc, 2),
            "main_language_est_accuracy_pct": round(est_acc, 2),
            "eng_token_precision_pct": round(eng_p * 100, 2),
            "eng_token_recall_pct": round(eng_r * 100, 2),
            "eng_token_f1_pct": round(eng_f1 * 100, 2),
            "cs_detection_recall_pct": round(cs_recall, 2),
            "composite_score_pct": round(composite, 2),
        })

    df_sentences = pd.DataFrame(sentence_rows)
    df_summary = pd.DataFrame(summary_rows).sort_values(
        by="composite_score_pct", ascending=False
    )
    return df_sentences, df_summary


# ---------------------------------------------------------------------------
# Exceli eksport
# ---------------------------------------------------------------------------


def export_report(
    df_sentences: pd.DataFrame,
    df_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Kirjuta tulemused Excelisse koos SentenceView + Summary lehtedega."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_sentences.to_excel(writer, sheet_name="SentenceView", index=False)
        df_summary.to_excel(writer, sheet_name="Summary", index=False)

        # Lisa joonealused märkused kokkuvõttetabeli alla
        ws = writer.sheets["Summary"]
        start_row = len(df_summary) + 3
        for i, (col, note) in enumerate(FOOTNOTES.items()):
            ws.cell(
                row=start_row + i, column=1, value=f"{col} – {note}"
            )

    print(f"  ✓ Saved: {output_path}")


# ---------------------------------------------------------------------------
# Põhi
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Run all language identifiers on chat_tsv + reddit data."
    )
    parser.add_argument(
        "--tsv-dir",
        type=str,
        default=str(ROOT_DIR / "data" / "chat" / "chat_tsv"),
        help="Path to chat_tsv directory",
    )
    parser.add_argument(
        "--reddit-xlsx",
        type=str,
        default=str(ROOT_DIR / "evaluation" / "reddit_margendus.xlsx"),
        help="Path to reddit_margendus.xlsx",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT_DIR / "evaluation" / "results"),
        help="Output directory for reports",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=1,
        help="Minimum sentence length in tokens (default: 1)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("FULL LANGUAGE IDENTIFIER REPORT")
    print("=" * 70)

    # 1. Registreeri tuvastajad
    print("\nRegistering detectors …")
    tools_token, tools_sequence = register_detectors()
    all_names = list(tools_sequence.keys()) + list(tools_token.keys())
    print(f"\n  Active detectors ({len(all_names)}): {', '.join(all_names)}\n")

    if not all_names:
        print("ERROR: No detectors available!")
        sys.exit(1)

    output_dir = Path(args.output_dir)

    # 2. Lae andmed
    print("Loading data …")
    chat_sentences = load_chat_tsv(Path(args.tsv_dir), min_words=args.min_words)
    reddit_sentences = load_reddit_xlsx(
        Path(args.reddit_xlsx), min_words=args.min_words
    )
    print()

    # 3. Hinda chat_tsv
    if chat_sentences:
        print("─" * 70)
        print(f"Evaluating chat_tsv ({len(chat_sentences)} sentences) …")
        print("─" * 70)
        df_sent, df_summ = evaluate_sentences(
            chat_sentences, tools_token, tools_sequence, "chat_tsv"
        )
        export_report(df_sent, df_summ, output_dir / "chat_tsv_full_report.xlsx")
        print("\n  Summary (chat_tsv):")
        print(df_summ.to_string(index=False))
        print()

    # 4. Hinda reddit
    if reddit_sentences:
        print("─" * 70)
        print(f"Evaluating reddit_margendus ({len(reddit_sentences)} sentences) …")
        print("─" * 70)
        df_sent, df_summ = evaluate_sentences(
            reddit_sentences, tools_token, tools_sequence, "reddit"
        )
        export_report(df_sent, df_summ, output_dir / "reddit_full_report.xlsx")
        print("\n  Summary (reddit):")
        print(df_summ.to_string(index=False))
        print()

    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()

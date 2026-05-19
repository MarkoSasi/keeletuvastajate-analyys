"""
Hinda keele tuvastajaid TSV-andmetel.

Kuldmärgendid tuletatakse `Margendatud_tekst` siltidest:
- tokenid siltide sees, mis sisaldavad `lang='eng'` => ENG
- kõik muud tokenid => EST
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from language_detector import LanguageDetector


LABELS = ("EST", "ENG", "MIXED")


@dataclass
class ToolStats:
    token_total: int = 0
    token_correct: int = 0
    eng_tp: int = 0
    eng_fp: int = 0
    eng_fn: int = 0
    sentence_total: int = 0
    sentence_tp: int = 0
    sentence_fp: int = 0
    sentence_fn: int = 0
    span_exact_tp: int = 0
    span_exact_pred: int = 0
    span_exact_gold: int = 0
    span_overlap_tp: int = 0
    span_overlap_pred: int = 0
    span_overlap_gold: int = 0
    mismatched_rows: int = 0


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def _safe_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    return text


def parse_annotated_tokens(annotated: str) -> Tuple[List[str], List[str]]:
    """
    TEKE-stiilis märgendatud tekst sõneloendiks ja kuldmärgenditeks.
    """
    tokens: List[str] = []
    labels: List[str] = []
    eng_depth = 0

    parts = re.findall(r"<[^>]+>|[^<]+", _safe_text(annotated))
    for part in parts:
        if part.startswith("<"):
            tag = part.lower()
            if tag == "</>":
                eng_depth = max(0, eng_depth - 1)
            else:
                if "lang='eng'" in tag or 'lang="eng"' in tag:
                    eng_depth += 1
            continue

        for tok in re.findall(r"[a-zA-ZäöüõÄÖÜÕšžŠŽ']+", part):
            tokens.append(tok)
            labels.append("ENG" if eng_depth > 0 else "EST")

    return tokens, labels


def label_spans(labels: List[str]) -> List[Tuple[str, int, int]]:
    """
    Teisenda sõnepõhised märgendid lõikudeks; jätab EST-lõigud välja.
    """
    spans: List[Tuple[str, int, int]] = []
    if not labels:
        return spans

    start = 0
    curr = labels[0]
    for i in range(1, len(labels)):
        if labels[i] != curr:
            if curr != "EST":
                spans.append((curr, start, i))
            start = i
            curr = labels[i]

    if curr != "EST":
        spans.append((curr, start, len(labels)))
    return spans


def spans_to_text(spans: List[Tuple[str, int, int]], tokens: List[str]) -> str:
    if not spans:
        return "-"
    parts = []
    for label, start, end in spans:
        frag = " ".join(tokens[start:end]).strip()
        if frag:
            parts.append(f"{frag} [{label}]")
    return " | ".join(parts) if parts else "-"


def _spans_overlap(a: Tuple[str, int, int], b: Tuple[str, int, int]) -> bool:
    return not (a[2] <= b[1] or b[2] <= a[1])


def exact_span_tp(
    gold_spans: List[Tuple[str, int, int]],
    pred_spans: List[Tuple[str, int, int]],
) -> int:
    return len(set(gold_spans).intersection(set(pred_spans)))


def overlap_span_tp(
    gold_spans: List[Tuple[str, int, int]],
    pred_spans: List[Tuple[str, int, int]],
) -> int:
    used_gold = set()
    tp = 0
    for pred in pred_spans:
        best_idx: Optional[int] = None
        best_overlap = -1
        for gi, gold in enumerate(gold_spans):
            if gi in used_gold:
                continue
            if pred[0] != gold[0]:
                continue
            if not _spans_overlap(pred, gold):
                continue
            overlap = min(pred[2], gold[2]) - max(pred[1], gold[1])
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = gi
        if best_idx is not None:
            used_gold.add(best_idx)
            tp += 1
    return tp


def map_rule_label(label: str) -> str:
    if label in ("ENG", "MIXED"):
        return label
    return "EST"


def evaluate(
    tsv_dir: str,
    output_path: str = "evaluation/tsv_identifier_report.xlsx",
    min_words: int = 5,
    pretrained_only: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    folder = Path(tsv_dir)
    files = sorted(folder.glob("*.tsv"))
    if not files:
        raise FileNotFoundError(f"No TSV files found in {folder}")

    detector = LanguageDetector()

    tools_token: Dict[str, Callable[[str], str]] = {}
    tools_sequence: Dict[str, Callable[[List[str]], List[str]]] = {}

    def _labels_from_char_spans(tokens: List[str], spans: List[Tuple[int, int]]) -> List[str]:
        labels = ["EST"] * len(tokens)
        if not tokens or not spans:
            return labels
        offsets: List[Tuple[int, int]] = []
        pos = 0
        for tok in tokens:
            start = pos
            end = start + len(tok)
            offsets.append((start, end))
            pos = end + 1  # üks eraldaja " ".join(tokens) sees
        for i, (tok_s, tok_e) in enumerate(offsets):
            for sp_s, sp_e in spans:
                if sp_e > tok_s and sp_s < tok_e:
                    labels[i] = "ENG"
                    break
        return labels

    def _find_phrase_spans(text: str, phrases: List[str]) -> List[Tuple[int, int]]:
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

    # Reeglipõhine jada-tööriist.
    def seq_rules(tokens: List[str]) -> List[str]:
        sentence = " ".join(tokens)
        predicted = detector.label_tokens(sentence)
        pred_tokens = [str(t["token"]).lower() for t in predicted]
        gold_tokens = [t.lower() for t in tokens]
        if len(predicted) == len(tokens) and pred_tokens == gold_tokens:
            return [map_rule_label(str(t["label"])) for t in predicted]
        return [map_rule_label(detector._classify_word_for_spans(tok)) for tok in tokens]

    tools_sequence["rules"] = seq_rules

    # Lingua (sõnatasand).
    try:
        from lingua import Language, LanguageDetectorBuilder

        ling = LanguageDetectorBuilder.from_languages(Language.ESTONIAN, Language.ENGLISH).build()

        def tok_lingua(word: str) -> str:
            if len(word) < 2:
                return "EST"
            lang = ling.detect_language_of(word)
            if lang == Language.ENGLISH:
                return "ENG"
            return "EST"

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
                    s_i = int(s)
                    e_i = int(e)
                    if e_i > s_i:
                        eng_spans.append((s_i, e_i))
                if eng_spans:
                    return _labels_from_char_spans(tokens, eng_spans)
            except Exception:
                pass
            return [tok_lingua(tok) for tok in tokens]

        tools_sequence["lingua-mixed"] = seq_lingua_mixed
    except Exception:
        pass

    # langid (sõnatasand).
    try:
        import langid

        langid.set_languages(["et", "en"])

        def tok_langid(word: str) -> str:
            if len(word) < 2:
                return "EST"
            lang, _ = langid.classify(word)
            return "ENG" if lang == "en" else "EST"

        tools_token["langid"] = tok_langid
    except Exception:
        pass

    # fasttext (sõnatasand).
    try:
        import fasttext

        ft_path = ROOT_DIR / "lid.176.bin"
        if ft_path.exists():
            ft_model = fasttext.load_model(str(ft_path))

            def tok_fasttext(word: str) -> str:
                if len(word) < 2:
                    return "EST"
                pred = ft_model.predict(word, k=1)[0][0].replace("__label__", "")
                return "ENG" if pred == "en" else "EST"

            tools_token["fasttext"] = tok_fasttext
    except Exception:
        pass

    # fast-langdetect (sõnatasand).
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
    except Exception:
        pass

    # MaskLID (sõna + koodivahetuse jada režiim).
    try:
        masklid_dir = ROOT_DIR / "evaluation" / "MaskLID"
        if str(masklid_dir) not in sys.path:
            sys.path.append(str(masklid_dir))
        from masklid import MaskLID

        masklid_model_path = ROOT_DIR / "lid.176.bin"
        if masklid_model_path.exists():
            masklid = MaskLID(str(masklid_model_path), languages=["__label__et", "__label__en"])

            def tok_masklid(word: str) -> str:
                if len(word) < 2:
                    return "EST"
                try:
                    labels, _ = masklid.predict(word, k=1)
                    if labels and len(labels) > 0:
                        top = str(labels[0]).replace("__label__", "")
                        return "ENG" if top == "en" else "EST"
                except Exception:
                    pass
                return "EST"

            def seq_masklid_cs(tokens: List[str]) -> List[str]:
                sentence = " ".join(tokens)
                try:
                    cs = masklid.predict_codeswitch(
                        sentence,
                        beta=10,
                        alpha=5,
                        min_prob=0.3,
                        min_length=1,
                    )
                except Exception:
                    return [tok_masklid(tok) for tok in tokens]
                eng_text = ""
                if isinstance(cs, dict):
                    eng_text = str(cs.get("__label__en", "") or cs.get("en", "")).strip()
                if not eng_text:
                    return [tok_masklid(tok) for tok in tokens]
                candidate_phrases = [p.strip() for p in re.split(r"[;|,]+", eng_text) if p.strip()]
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
    except Exception:
        pass

    # AnE-LID (HF sõneklassifikaator).
    try:
        from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

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
                group = str(ent.get("entity_group", ent.get("entity", ""))).lower()
                if "english" in group and "not" not in group:
                    s = int(ent.get("start", -1))
                    e = int(ent.get("end", -1))
                    if e > s >= 0:
                        spans.append((s, e))
            return _labels_from_char_spans(tokens, spans)

        tools_sequence["ane-lid"] = seq_ane_lid
    except Exception:
        pass

    if pretrained_only:
        tools_sequence = {k: v for k, v in tools_sequence.items() if k not in {"rules", "viterbi"}}
    all_tools = {**tools_sequence, **tools_token}
    if not all_tools:
        raise ValueError("No identifiers available after filtering")
    stats = {name: ToolStats() for name in all_tools}
    details: List[Dict[str, object]] = []
    sentence_rows: List[Dict[str, object]] = []

    processed_rows = 0
    skipped_short_rows = 0
    total_files = len(files)
    for idx, file in enumerate(files, 1):
        df = pd.read_csv(file, sep="\t", dtype=str, keep_default_na=False, encoding="utf-8")
        for row_idx, row in df.iterrows():
            annotated = _safe_text(row.get("Margendatud_tekst", ""))
            if not annotated:
                continue

            tokens, gold_labels = parse_annotated_tokens(annotated)
            if not tokens:
                continue
            if len(tokens) < min_words:
                skipped_short_rows += 1
                continue

            processed_rows += 1
            sentence = " ".join(tokens)
            gold_spans = label_spans(gold_labels)
            gold_cs = any(lbl != "EST" for lbl in gold_labels)
            sentence_row: Dict[str, object] = {
                "file": file.name,
                "row": int(row_idx),
                "sentence": sentence,
                "gold_sentence_label": "CS" if gold_cs else "EST",
                "gold_cs_parts": spans_to_text(gold_spans, tokens),
            }

            for tool_name, tool in tools_sequence.items():
                tool_stat = stats[tool_name]
                pred_labels = tool(tokens)
                if len(pred_labels) != len(gold_labels):
                    tool_stat.mismatched_rows += 1
                    sentence_row[f"{tool_name}_sentence_label"] = "MISMATCH"
                    sentence_row[f"{tool_name}_token_correct_pct"] = 0.0
                    sentence_row[f"{tool_name}_cs_detection_correct_pct"] = 0.0
                    sentence_row[f"{tool_name}_cs_parts"] = "-"
                    continue
                _accumulate_metrics(
                    tool_stat,
                    gold_labels,
                    pred_labels,
                    gold_spans,
                    tool_name,
                    file.name,
                    int(row_idx),
                    sentence,
                    details,
                    gold_cs,
                )
                pred_spans = label_spans(pred_labels)
                pred_cs = any(lbl != "EST" for lbl in pred_labels)
                token_correct = sum(1 for g, p in zip(gold_labels, pred_labels) if g == p)
                token_correct_pct = 100.0 * token_correct / len(gold_labels)

                sentence_row[f"{tool_name}_sentence_label"] = "CS" if pred_cs else "EST"
                sentence_row[f"{tool_name}_token_correct_pct"] = round(token_correct_pct, 2)
                sentence_row[f"{tool_name}_cs_detection_correct_pct"] = (
                    100.0 if pred_cs == gold_cs else 0.0
                )
                sentence_row[f"{tool_name}_cs_parts"] = spans_to_text(pred_spans, tokens)

            for tool_name, tok_tool in tools_token.items():
                tool_stat = stats[tool_name]
                pred_labels = [tok_tool(tok) for tok in tokens]
                _accumulate_metrics(
                    tool_stat,
                    gold_labels,
                    pred_labels,
                    gold_spans,
                    tool_name,
                    file.name,
                    int(row_idx),
                    sentence,
                    details,
                    gold_cs,
                )
                pred_spans = label_spans(pred_labels)
                pred_cs = any(lbl != "EST" for lbl in pred_labels)
                token_correct = sum(1 for g, p in zip(gold_labels, pred_labels) if g == p)
                token_correct_pct = 100.0 * token_correct / len(gold_labels)

                sentence_row[f"{tool_name}_sentence_label"] = "CS" if pred_cs else "EST"
                sentence_row[f"{tool_name}_token_correct_pct"] = round(token_correct_pct, 2)
                sentence_row[f"{tool_name}_cs_detection_correct_pct"] = (
                    100.0 if pred_cs == gold_cs else 0.0
                )
                sentence_row[f"{tool_name}_cs_parts"] = spans_to_text(pred_spans, tokens)

            sentence_rows.append(sentence_row)

        if idx % 15 == 0 or idx == total_files:
            print(f"Processed {idx}/{total_files} files...")

    summary_rows = []
    for tool_name, st in stats.items():
        token_acc = st.token_correct / st.token_total if st.token_total else 0.0
        eng_p, eng_r, eng_f1 = _prf(st.eng_tp, st.eng_fp, st.eng_fn)
        cs_p, cs_r, cs_f1 = _prf(st.sentence_tp, st.sentence_fp, st.sentence_fn)

        exact_p = st.span_exact_tp / st.span_exact_pred if st.span_exact_pred else 0.0
        exact_r = st.span_exact_tp / st.span_exact_gold if st.span_exact_gold else 0.0
        exact_f1 = (
            2 * exact_p * exact_r / (exact_p + exact_r) if (exact_p + exact_r) else 0.0
        )
        ov_p = st.span_overlap_tp / st.span_overlap_pred if st.span_overlap_pred else 0.0
        ov_r = st.span_overlap_tp / st.span_overlap_gold if st.span_overlap_gold else 0.0
        ov_f1 = 2 * ov_p * ov_r / (ov_p + ov_r) if (ov_p + ov_r) else 0.0

        summary_rows.append(
            {
                "tool": tool_name,
                "rows_processed": processed_rows,
                "rows_skipped_short": skipped_short_rows,
                "min_words": min_words,
                "pretrained_only": pretrained_only,
                "token_total": st.token_total,
                "token_accuracy": token_acc,
                "eng_precision": eng_p,
                "eng_recall": eng_r,
                "eng_f1": eng_f1,
                "sentence_cs_precision": cs_p,
                "sentence_cs_recall": cs_r,
                "sentence_cs_f1": cs_f1,
                "span_exact_f1": exact_f1,
                "span_overlap_f1": ov_f1,
                "mismatched_rows": st.mismatched_rows,
            }
        )

    df_summary = pd.DataFrame(summary_rows).sort_values(by="span_overlap_f1", ascending=False)
    df_detail = pd.DataFrame(details)
    df_sentence = pd.DataFrame(sentence_rows)

    if not df_sentence.empty:
        ordered_cols = [
            "file",
            "row",
            "sentence",
            "gold_sentence_label",
            "gold_cs_parts",
        ]
        for tool_name in all_tools.keys():
            ordered_cols.extend(
                [
                    f"{tool_name}_sentence_label",
                    f"{tool_name}_token_correct_pct",
                    f"{tool_name}_cs_detection_correct_pct",
                    f"{tool_name}_cs_parts",
                ]
            )
        df_sentence = df_sentence[[c for c in ordered_cols if c in df_sentence.columns]]

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
        df_sentence.to_excel(writer, sheet_name="SentenceView", index=False)
        df_detail.to_excel(writer, sheet_name="ErrorsSample", index=False)

    return df_summary, df_sentence


def _accumulate_metrics(
    st: ToolStats,
    gold_labels: List[str],
    pred_labels: List[str],
    gold_spans: List[Tuple[str, int, int]],
    tool_name: str,
    file_name: str,
    row_idx: int,
    sentence: str,
    details: List[Dict[str, object]],
    gold_cs: bool,
) -> None:
    st.token_total += len(gold_labels)
    for g, p in zip(gold_labels, pred_labels):
        if g == p:
            st.token_correct += 1
        if p == "ENG" and g == "ENG":
            st.eng_tp += 1
        elif p == "ENG" and g != "ENG":
            st.eng_fp += 1
        elif p != "ENG" and g == "ENG":
            st.eng_fn += 1

    st.sentence_total += 1
    pred_cs = any(lbl != "EST" for lbl in pred_labels)
    if pred_cs and gold_cs:
        st.sentence_tp += 1
    elif pred_cs and not gold_cs:
        st.sentence_fp += 1
    elif (not pred_cs) and gold_cs:
        st.sentence_fn += 1

    pred_spans = label_spans(pred_labels)
    st.span_exact_gold += len(gold_spans)
    st.span_exact_pred += len(pred_spans)
    st.span_exact_tp += exact_span_tp(gold_spans, pred_spans)
    st.span_overlap_gold += len(gold_spans)
    st.span_overlap_pred += len(pred_spans)
    st.span_overlap_tp += overlap_span_tp(gold_spans, pred_spans)

    if len(details) < 600 and (gold_spans != pred_spans):
        details.append(
            {
                "tool": tool_name,
                "file": file_name,
                "row": row_idx,
                "sentence": sentence,
                "gold_labels": " ".join(gold_labels),
                "pred_labels": " ".join(pred_labels),
                "gold_spans": " | ".join(f"{l}:{s}-{e}" for l, s, e in gold_spans),
                "pred_spans": " | ".join(f"{l}:{s}-{e}" for l, s, e in pred_spans),
            }
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run language identifiers on annotated TSV corpus.")
    parser.add_argument(
        "--tsv-dir",
        type=str,
        default=str(ROOT_DIR / "data" / "chat" / "chat_tsv"),
        help="Path to directory containing TSV files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT_DIR / "evaluation" / "results" / "tsv_identifier_report.xlsx"),
        help="Output Excel report path",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=5,
        help="Minimum sentence length in words to include in analysis (default: 5)",
    )
    parser.add_argument(
        "--pretrained-only",
        action="store_true",
        help="Use only pretrained identifiers (exclude project-specific rules/viterbi).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary, _ = evaluate(
        tsv_dir=args.tsv_dir,
        output_path=args.output,
        min_words=args.min_words,
        pretrained_only=args.pretrained_only,
    )
    print("\nTSV identifier evaluation complete:")
    print(summary.to_string(index=False))

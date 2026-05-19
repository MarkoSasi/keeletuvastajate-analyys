
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from utils import TOKEN_RE


def tokenize_words(text: str) -> List[str]:
    return TOKEN_RE.findall(text)


def normalize_sentence(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\säöüõšž]", " ", text)
    return " ".join(text.split())


def group_eng_spans(tokens: List[str], labels: List[str]) -> List[str]:
    spans: List[str] = []
    current: List[str] = []
    for tok, lbl in zip(tokens, labels):
        if lbl == "ENG":
            current.append(tok)
        else:
            if current:
                spans.append(" ".join(current))
                current = []
    if current:
        spans.append(" ".join(current))
    return spans


def get_main_language(labels: List[str]) -> str:
    if not labels:
        return "UNK"
    c = Counter(labels)
    est = c.get("EST", 0)
    eng = c.get("ENG", 0)
    mixed = c.get("MIXED", 0)
    if mixed > max(est, eng):
        return "MIXED"
    if est >= eng:
        return "EST"
    return "ENG"


class IdentifierSuite:

    def __init__(self, detector):
        self.detector = detector
        self.token_tools: Dict[str, Callable[[str], str]] = {}
        self.sequence_tools: Dict[str, Callable[[str, List[str], List[Tuple[int, int]]], List[str]]] = {}
        self.sentence_tools: Dict[str, Callable[[str], str]] = {}
        self._register_tools()

    # ── Abifunktsioonid ───────────────────────────────────────

    @staticmethod
    def _labels_from_char_spans(
        token_offsets: List[Tuple[int, int]],
        spans: List[Tuple[int, int]],
    ) -> List[str]:
        labels = ["EST"] * len(token_offsets)
        if not token_offsets or not spans:
            return labels
        for i, (tok_s, tok_e) in enumerate(token_offsets):
            for sp_s, sp_e in spans:
                if sp_e > tok_s and sp_s < tok_e:
                    labels[i] = "ENG"
                    break
        return labels

    @staticmethod
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

    def _register_tools(self) -> None:
        self._register_rules()
        self._register_lingua()
        self._register_langid()
        self._register_fasttext()
        self._register_fast_langdetect()
        self._register_masklid()
        self._register_ane_lid()

    def _register_rules(self) -> None:
        def seq_rules(sentence: str) -> List[str]:
            token_info = self.detector.label_tokens(sentence)
            return [
                "MIXED" if str(t["label"]) == "MIXED"
                else "ENG" if str(t["label"]) == "ENG"
                else "EST"
                for t in token_info
            ]

        self.token_tools["rules"] = lambda word: (
            "ENG" if self.detector._classify_word_for_spans(word) == "ENG" else "EST"
        )
        self.sentence_tools["rules"] = lambda s: get_main_language(seq_rules(s))

    def _register_lingua(self) -> None:
        try:
            from lingua import Language, LanguageDetectorBuilder

            ling = LanguageDetectorBuilder.from_languages(Language.ESTONIAN, Language.ENGLISH).build()

            def tok_lingua(word: str) -> str:
                if len(word) < 2:
                    return "EST"
                lang = ling.detect_language_of(word)
                return "ENG" if lang == Language.ENGLISH else "EST"

            def sent_lingua(sentence: str) -> str:
                lang = ling.detect_language_of(sentence)
                if lang == Language.ENGLISH:
                    return "ENG"
                if lang == Language.ESTONIAN:
                    return "EST"
                return "UNK"

            self.token_tools["lingua"] = tok_lingua
            self.sentence_tools["lingua"] = sent_lingua

            def seq_lingua_mixed(
                sentence: str,
                tokens: List[str],
                offsets: List[Tuple[int, int]],
            ) -> List[str]:
                if not tokens:
                    return []
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
                        return self._labels_from_char_spans(offsets, eng_spans)
                except Exception:
                    pass
                return [tok_lingua(tok) for tok in tokens]

            self.sequence_tools["lingua-mixed"] = seq_lingua_mixed
        except Exception:
            pass

    def _register_langid(self) -> None:
        try:
            import langid

            langid.set_languages(["et", "en"])

            def tok_langid(word: str) -> str:
                if len(word) < 2:
                    return "EST"
                lang, _ = langid.classify(word)
                return "ENG" if lang == "en" else "EST"

            def sent_langid(sentence: str) -> str:
                lang, _ = langid.classify(sentence)
                return "ENG" if lang == "en" else "EST" if lang == "et" else "UNK"

            self.token_tools["langid"] = tok_langid
            self.sentence_tools["langid"] = sent_langid
        except Exception:
            pass

    def _register_fasttext(self) -> None:
        try:
            import fasttext

            model_path = Path("models/lid.176.bin")
            if not model_path.exists():
                model_path = Path("lid.176.bin")
            if model_path.exists():
                ft = fasttext.load_model(str(model_path))

                def tok_ft(word: str) -> str:
                    if len(word) < 2:
                        return "EST"
                    pred = ft.predict(word, k=1)[0][0].replace("__label__", "")
                    return "ENG" if pred == "en" else "EST"

                def sent_ft(sentence: str) -> str:
                    pred = ft.predict(sentence, k=1)[0][0].replace("__label__", "")
                    return "ENG" if pred == "en" else "EST" if pred == "et" else "UNK"

                self.token_tools["fasttext"] = tok_ft
                self.sentence_tools["fasttext"] = sent_ft
        except Exception:
            pass

    def _register_fast_langdetect(self) -> None:
        try:
            from fast_langdetect import detect as fld_detect

            def tok_fld(word: str) -> str:
                if len(word) < 2:
                    return "EST"
                try:
                    res = fld_detect(word)
                    if res and len(res) > 0:
                        lang = str(res[0].get("lang", "et"))
                        return "ENG" if lang in ("en", "eng") else "EST"
                except Exception:
                    pass
                return "EST"

            def sent_fld(sentence: str) -> str:
                try:
                    res = fld_detect(sentence)
                    if res and len(res) > 0:
                        lang = str(res[0].get("lang", "et"))
                        return "ENG" if lang in ("en", "eng") else "EST" if lang in ("et", "est") else "UNK"
                except Exception:
                    pass
                return "UNK"

            self.token_tools["fast-langdetect"] = tok_fld
            self.sentence_tools["fast-langdetect"] = sent_fld
        except Exception:
            pass

    def _register_masklid(self) -> None:
        try:
            masklid_dir = Path("evaluation") / "MaskLID"
            if str(masklid_dir) not in sys.path:
                sys.path.append(str(masklid_dir))
            from masklid import MaskLID

            model_path = Path("models/lid.176.bin")
            if not model_path.exists():
                model_path = Path("lid.176.bin")
            if model_path.exists():
                masklid = MaskLID(str(model_path), languages=["__label__et", "__label__en"])

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

                def sent_masklid(sentence: str) -> str:
                    try:
                        labels, _ = masklid.predict(sentence, k=1)
                        if labels and len(labels) > 0:
                            top = str(labels[0]).replace("__label__", "")
                            if top == "en":
                                return "ENG"
                            if top == "et":
                                return "EST"
                    except Exception:
                        pass
                    return "UNK"

                def seq_masklid_cs(
                    sentence: str,
                    tokens: List[str],
                    offsets: List[Tuple[int, int]],
                ) -> List[str]:
                    if not tokens:
                        return []
                    try:
                        cs = masklid.predict_codeswitch(
                            sentence, beta=10, alpha=5, min_prob=0.3, min_length=1,
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
                    spans = self._find_phrase_spans(sentence, candidate_phrases)
                    if not spans:
                        spans = self._find_phrase_spans(sentence, [eng_text])
                    if not spans:
                        return [tok_masklid(tok) for tok in tokens]
                    return self._labels_from_char_spans(offsets, spans)

                self.token_tools["masklid"] = tok_masklid
                self.sentence_tools["masklid"] = sent_masklid
                self.sequence_tools["masklid-cs"] = seq_masklid_cs
        except Exception:
            pass

    def _register_ane_lid(self) -> None:
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

            def seq_ane_lid(
                sentence: str,
                tokens: List[str],
                offsets: List[Tuple[int, int]],
            ) -> List[str]:
                if not tokens:
                    return []
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
                return self._labels_from_char_spans(offsets, spans)

            self.sequence_tools["ane-lid"] = seq_ane_lid
        except Exception:
            pass

    # ── Järeldus ───────────────────────────────────────────────────────

    def analyze_sentence(self, sentence: str) -> Dict[str, Dict[str, object]]:
        token_matches = list(TOKEN_RE.finditer(sentence))
        tokens = [m.group(0) for m in token_matches]
        offsets = [(m.start(), m.end()) for m in token_matches]
        out: Dict[str, Dict[str, object]] = {}
        for name in self.available_tools():
            if name in self.sequence_tools:
                labels = self.sequence_tools[name](sentence, tokens, offsets)
            else:
                tok_fn = self.token_tools[name]
                labels = [tok_fn(tok) for tok in tokens]

            if len(labels) != len(tokens):
                labels = labels[: len(tokens)] + (["EST"] * max(0, len(tokens) - len(labels)))
            labels = [lbl if lbl in {"EST", "ENG", "MIXED"} else "EST" for lbl in labels]
            cs_spans = group_eng_spans(tokens, labels)
            main = (
                self.sentence_tools[name](sentence)
                if name in self.sentence_tools
                else get_main_language(labels)
            )
            out[name] = {
                "main": main,
                "labels": labels,
                "cs_spans": cs_spans,
                "eng_count": sum(1 for lbl in labels if lbl == "ENG"),
            }
        return out

    def available_tools(self) -> List[str]:
        return sorted(set(self.token_tools.keys()).union(self.sequence_tools.keys()))

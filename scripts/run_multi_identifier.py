#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import re
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from config import REDDIT_CATEGORIES, SEARCH_KEYWORDS, SUBREDDITS, TIME_FILTERS
from scraper import RedditScraper
from sentence_processor import SentenceProcessor
from language_detector import LanguageDetector
from identifier_suite import IdentifierSuite


TOKEN_RE = re.compile(r"[a-zA-ZäöüõÄÖÜÕšžŠŽ']+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect r/Eesti code-switching sentences and compare identifiers."
    )
    parser.add_argument("--limit", type=int, default=500, help="Target sentence count (default: 500)")
    parser.add_argument(
        "--output",
        type=str,
        default="eesti_reddit_codeswitch_identifiers.xlsx",
        help="Output xlsx path",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=5,
        help="Minimum sentence length in words (default: 5)",
    )
    parser.add_argument(
        "--min-eng-words",
        type=int,
        default=2,
        help="Minimum ENG words for code-switch acceptance (default: 2)",
    )
    parser.add_argument(
        "--allow-eng-dominant",
        action="store_true",
        help="Allow ENG-dominant mixed sentences (default: disabled; only EST-dominant kept).",
    )
    parser.add_argument(
        "--max-posts-per-batch",
        type=int,
        default=500,
        help="Max posts per category/time batch (default: 500)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint and keep saving progress.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Checkpoint JSON path (default: <output>.checkpoint.json).",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=20,
        help="Save checkpoint after every N newly kept sentences (default: 20).",
    )
    parser.add_argument(
        "--max-runtime-min",
        type=int,
        default=0,
        help="Stop collect cycle after N minutes (0 = unlimited).",
    )
    parser.add_argument(
        "--max-restarts",
        type=int,
        default=0,
        help="Auto-restart collect cycles up to N times when limit is not yet reached (0 = no restarts).",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Only collect/resume sentences and save checkpoint; skip annotation/export scoring.",
    )
    parser.add_argument(
        "--require-pretrained-confirmation",
        action="store_true",
        help="During collection, require selected pretrained identifiers to confirm an ENG span.",
    )
    parser.add_argument(
        "--pretrained-confirm-tools",
        type=str,
        default="ane-lid,lingua-mixed",
        help="Comma-separated pretrained tools for collection confirmation (default: ane-lid,lingua-mixed).",
    )
    parser.add_argument(
        "--pretrained-confirm-min-tools",
        type=int,
        default=1,
        help="Minimum number of selected pretrained tools that must confirm (default: 1).",
    )
    parser.add_argument(
        "--pretrained-confirm-min-eng-words",
        type=int,
        default=2,
        help="Minimum ENG token count predicted by a confirming pretrained tool (default: 2).",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def tokenize_words(text: str) -> List[str]:
    return TOKEN_RE.findall(text)


def normalize_sentence(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\säöüõšž]", " ", text)
    return " ".join(text.split())


def group_eng_spans(tokens: List[str], labels: List[str]) -> List[str]:
    spans = []
    current = []
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


class RedditCodeSwitchPipeline:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.scraper = RedditScraper()
        self.processor = SentenceProcessor()
        self.detector = LanguageDetector()
        needs_identifiers = (not self.args.collect_only) or bool(self.args.require_pretrained_confirmation)
        self.identifiers = IdentifierSuite(self.detector) if needs_identifiers else None
        self.pretrained_confirm_tools = [
            t.strip() for t in str(getattr(self.args, "pretrained_confirm_tools", "")).split(",") if t.strip()
        ]
        self.pretrained_confirm_min_tools = max(1, int(getattr(self.args, "pretrained_confirm_min_tools", 1)))
        self.pretrained_confirm_min_eng_words = max(
            1, int(getattr(self.args, "pretrained_confirm_min_eng_words", self.args.min_eng_words))
        )
        self._pretrained_confirm_available: List[str] = []
        if self.args.require_pretrained_confirmation:
            if self.identifiers is None:
                raise RuntimeError("Pretrained confirmation requested but identifiers are not initialized")
            available = set(self.identifiers.available_tools())
            self._pretrained_confirm_available = [t for t in self.pretrained_confirm_tools if t in available]
            if not self._pretrained_confirm_available:
                raise RuntimeError(
                    "No requested pretrained confirmation tools available. "
                    f"Requested={self.pretrained_confirm_tools}, available={sorted(available)}"
                )
            self.log(
                "Pretrained confirmation enabled: tools="
                f"{self._pretrained_confirm_available}, min_tools={self.pretrained_confirm_min_tools}, "
                f"min_eng_words={self.pretrained_confirm_min_eng_words}"
            )
        self.rows: List[Dict[str, object]] = []
        self.seen = set()
        self.visited_posts = set()
        self.stats = {
            "posts_scraped": 0,
            "comments_scraped": 0,
            "sentences_processed": 0,
            "candidates_kept": 0,
            "restarts": 0,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        self.checkpoint_path = (
            Path(args.checkpoint) if str(args.checkpoint).strip() else Path(f"{args.output}.checkpoint.json")
        )
        self._last_checkpoint_rows = 0
        self._ops_since_checkpoint = 0
        if self.args.resume:
            self._restore_progress()

    def _pretrained_confirms_codeswitch(self, sentence: str) -> bool:
        if not self.args.require_pretrained_confirmation:
            return True
        if self.identifiers is None:
            return False

        result = self.identifiers.analyze_sentence(sentence)
        confirmations = 0
        for tool in self._pretrained_confirm_available:
            tool_res = result.get(tool) or {}
            labels = [str(lbl) for lbl in (tool_res.get("labels") or [])]
            if not labels:
                continue
            eng_count = sum(1 for lbl in labels if lbl == "ENG")
            est_count = sum(1 for lbl in labels if lbl == "EST")
            if eng_count < self.pretrained_confirm_min_eng_words:
                continue
            if est_count < 2:
                continue
            confirmations += 1
            if confirmations >= self.pretrained_confirm_min_tools:
                return True
        return False

    def log(self, msg: str):
        if self.args.verbose:
            print(msg)

    def _checkpoint_payload(self) -> Dict[str, object]:
        return {
            "rows": self.rows,
            "seen": sorted(self.seen),
            "visited_posts": sorted(self.visited_posts),
            "stats": self.stats,
            "meta": {
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "output": str(self.args.output),
                "limit": int(self.args.limit),
            },
        }

    def _save_checkpoint(self, force: bool = False) -> None:
        if not self.args.resume:
            return
        if not force:
            self._ops_since_checkpoint += 1
            rows_delta = len(self.rows) - self._last_checkpoint_rows
            # Salvesta kas lausete kasvu või perioodilise operatsiooniloendi alusel.
            if rows_delta < max(1, int(self.args.save_every)) and self._ops_since_checkpoint < 50:
                return
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.checkpoint_path.with_suffix(self.checkpoint_path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._checkpoint_payload(), ensure_ascii=True), encoding="utf-8")
        tmp.replace(self.checkpoint_path)
        self._last_checkpoint_rows = len(self.rows)
        self._ops_since_checkpoint = 0

    def _restore_progress(self) -> None:
        # 1. prioriteet: checkpoint json
        if self.checkpoint_path.exists():
            try:
                payload = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
                rows = payload.get("rows", [])
                self.rows = [
                    {
                        "sentence": str(r.get("sentence", "")),
                        "source_url": str(r.get("source_url", "")),
                        "source_type": str(r.get("source_type", "")),
                    }
                    for r in rows
                    if str(r.get("sentence", "")).strip()
                ]
                self.seen = set(str(x) for x in payload.get("seen", []))
                if not self.seen:
                    self.seen = {normalize_sentence(r["sentence"]) for r in self.rows if r.get("sentence")}
                self.visited_posts = set(str(x) for x in payload.get("visited_posts", []))
                saved_stats = payload.get("stats", {})
                if isinstance(saved_stats, dict):
                    for k in self.stats:
                        if k in saved_stats:
                            self.stats[k] = saved_stats[k]
                self.stats["candidates_kept"] = len(self.rows)
                self._last_checkpoint_rows = len(self.rows)
                print(f"Resumed checkpoint: {len(self.rows)} sentences, {len(self.visited_posts)} visited posts")
                return
            except Exception as exc:
                print(f"Warning: failed to load checkpoint {self.checkpoint_path}: {exc}")

        # 2. prioriteet: olemasolev väljund-xlsx Sentences leht
        out = Path(self.args.output)
        if out.exists():
            try:
                df = pd.read_excel(out, sheet_name="Sentences")
                required = {"sentence", "source_url", "source_type"}
                if required.issubset(df.columns):
                    restored = []
                    for _, row in df.iterrows():
                        sent = str(row.get("sentence", "")).strip()
                        if not sent:
                            continue
                        restored.append(
                            {
                                "sentence": sent,
                                "source_url": str(row.get("source_url", "")),
                                "source_type": str(row.get("source_type", "")),
                            }
                        )
                    self.rows = restored
                    self.seen = {normalize_sentence(r["sentence"]) for r in self.rows}
                    self.stats["candidates_kept"] = len(self.rows)
                    self._last_checkpoint_rows = len(self.rows)
                    print(f"Resumed from output workbook: {len(self.rows)} sentences")
            except Exception as exc:
                print(f"Warning: failed to restore from output workbook {out}: {exc}")

    def _is_potential_codeswitch(self, sentence: str) -> bool:
        if len(tokenize_words(sentence)) < self.args.min_words:
            return False

        # Säilita esmalt vaid range koodivahetuse kandidaadid.
        if not self.detector.is_codeswitched(sentence, min_english_words=self.args.min_eng_words):
            return False

        # Jõusta lauseülene domineerimise reegel: eesti on tokeniarvult peakeel.
        token_info = self.detector.label_tokens(sentence)
        est_words = sum(1 for t in token_info if str(t.get("label")) == "EST")
        eng_words = sum(1 for t in token_info if str(t.get("label")) == "ENG")
        if eng_words < self.args.min_eng_words or est_words < 2:
            return False
        if not self.args.allow_eng_dominant and est_words <= eng_words:
            return False
        if not self._pretrained_confirms_codeswitch(sentence):
            return False
        return True

    def _add_sentence(self, sentence: str, source_url: str, source_type: str) -> None:
        norm = normalize_sentence(sentence)
        if not norm or norm in self.seen:
            return
        if not self._is_potential_codeswitch(sentence):
            return

        self.seen.add(norm)
        self.rows.append(
            {
                "sentence": sentence,
                "source_url": source_url,
                "source_type": source_type,
            }
        )
        self.stats["candidates_kept"] += 1
        self._save_checkpoint(force=False)

    def _process_text(self, text: str, source_url: str, source_type: str) -> None:
        if not text:
            return
        sentences = self.processor.process_text(text)
        self.stats["sentences_processed"] += len(sentences)
        for sent in sentences:
            if len(self.rows) >= self.args.limit:
                return
            self._add_sentence(sent, source_url, source_type)

    def _scrape_post(self, post: dict) -> None:
        post_id = str(post.get("id", "")).strip()
        if post_id:
            if post_id in self.visited_posts:
                return
            self.visited_posts.add(post_id)
        src = post.get("url", "")
        self._process_text(post.get("title", ""), src, "post_title")
        self._process_text(post.get("selftext", ""), src, "post_body")
        if len(self.rows) >= self.args.limit:
            return
        if post.get("num_comments", 0) > 0:
            comments = self.scraper.fetch_post_comments(post.get("permalink", ""))
            self.stats["comments_scraped"] += len(comments)
            for c in comments:
                if len(self.rows) >= self.args.limit:
                    return
                self._process_text(c, src, "comment")

    def collect(self) -> bool:
        deadline = None
        if int(self.args.max_runtime_min) > 0:
            deadline = time.time() + int(self.args.max_runtime_min) * 60

        pbar = tqdm(total=self.args.limit, desc="CS Sentences", unit="sent", ncols=90)
        pbar.n = len(self.rows)
        pbar.refresh()
        try:
            for subreddit in SUBREDDITS:
                if len(self.rows) >= self.args.limit:
                    break
                if deadline is not None and time.time() >= deadline:
                    break
                self.log(f"\nSubreddit: r/{subreddit}")
                for category in REDDIT_CATEGORIES:
                    if len(self.rows) >= self.args.limit:
                        break
                    if deadline is not None and time.time() >= deadline:
                        break
                    time_filters = TIME_FILTERS if category in {"top", "controversial"} else ["all"]
                    for tf in time_filters:
                        if len(self.rows) >= self.args.limit:
                            break
                        if deadline is not None and time.time() >= deadline:
                            break
                        posts = self.scraper.fetch_all_posts(
                            category=category,
                            max_posts=self.args.max_posts_per_batch,
                            time_filter=tf,
                            subreddit=subreddit,
                        )
                        self.stats["posts_scraped"] += len(posts)
                        for post in posts:
                            if len(self.rows) >= self.args.limit:
                                break
                            if deadline is not None and time.time() >= deadline:
                                break
                            before = len(self.rows)
                            self._scrape_post(post)
                            if len(self.rows) > before:
                                pbar.n = len(self.rows)
                                pbar.refresh()
                            self._save_checkpoint(force=False)

            # Märksõnafaas
            if len(self.rows) < self.args.limit:
                self.log("\nKeyword search phase...")
                for _, keywords in SEARCH_KEYWORDS.items():
                    if deadline is not None and time.time() >= deadline:
                        break
                    for kw in keywords:
                        if len(self.rows) >= self.args.limit:
                            break
                        if deadline is not None and time.time() >= deadline:
                            break
                        posts = self.scraper.search_subreddit(kw, limit=100, subreddit=SUBREDDITS[0])
                        self.stats["posts_scraped"] += len(posts)
                        for post in posts:
                            if len(self.rows) >= self.args.limit:
                                break
                            if deadline is not None and time.time() >= deadline:
                                break
                            before = len(self.rows)
                            self._scrape_post(post)
                            if len(self.rows) > before:
                                pbar.n = len(self.rows)
                                pbar.refresh()
                            self._save_checkpoint(force=False)
                    if len(self.rows) >= self.args.limit:
                        break
        finally:
            pbar.close()
            self._save_checkpoint(force=True)
        return len(self.rows) >= self.args.limit

    def annotate(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.identifiers is None:
            raise RuntimeError("Identifiers are not initialized in collect-only mode")
        tools = self.identifiers.available_tools()
        records = []

        per_tool = defaultdict(lambda: {"main_agree": 0, "count": 0, "tp": 0, "fp": 0, "fn": 0, "cs_nonempty": 0})

        for i, row in enumerate(self.rows, 1):
            sentence = str(row["sentence"])
            tokens = tokenize_words(sentence)
            result = self.identifiers.analyze_sentence(sentence)
            mains = [result[t]["main"] for t in tools if result[t]["main"] in {"EST", "ENG", "MIXED"}]
            consensus_main = Counter(mains).most_common(1)[0][0] if mains else "UNK"

            # Ankrutoken-id ENG-jaoks reeglipõhise tuvastaja põhjal.
            if "rules" in result:
                anchor = [1 if lbl == "ENG" else 0 for lbl in result["rules"]["labels"]]
            else:
                anchor = [0] * len(tokens)

            rec = {
                "id": i,
                "sentence": sentence,
                "source_url": row["source_url"],
                "source_type": row["source_type"],
                "word_count": len(tokens),
                "anchor_cs_parts": " | ".join(group_eng_spans(tokens, ["ENG" if a else "EST" for a in anchor])) or "-",
                "consensus_main": consensus_main,
            }

            for tool in tools:
                main = result[tool]["main"]
                cs_spans = result[tool]["cs_spans"]
                labels = result[tool]["labels"]
                pred = [1 if lbl == "ENG" else 0 for lbl in labels]
                tp = sum(1 for p, a in zip(pred, anchor) if p == 1 and a == 1)
                fp = sum(1 for p, a in zip(pred, anchor) if p == 1 and a == 0)
                fn = sum(1 for p, a in zip(pred, anchor) if p == 0 and a == 1)
                prec = tp / (tp + fp) if (tp + fp) else 0.0
                rec_score = tp / (tp + fn) if (tp + fn) else 0.0
                f1 = (2 * prec * rec_score / (prec + rec_score)) if (prec + rec_score) else 0.0

                rec[f"{tool}_main"] = main
                rec[f"{tool}_cs_parts"] = " | ".join(cs_spans) if cs_spans else "-"
                rec[f"{tool}_token_f1_vs_anchor"] = round(f1 * 100, 2)

                s = per_tool[tool]
                s["count"] += 1
                s["main_agree"] += 1 if main == consensus_main else 0
                s["tp"] += tp
                s["fp"] += fp
                s["fn"] += fn
                s["cs_nonempty"] += 1 if cs_spans else 0

            records.append(rec)

        df = pd.DataFrame(records)

        score_rows = []
        for tool in tools:
            s = per_tool[tool]
            n = max(s["count"], 1)
            main_agree = s["main_agree"] / n
            cs_rate = s["cs_nonempty"] / n
            p = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) else 0.0
            r = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) else 0.0
            f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
            composite = 0.4 * main_agree + 0.6 * f1
            score_rows.append(
                {
                    "identifier": tool,
                    "sentences": s["count"],
                    "main_language_agreement_pct": round(main_agree * 100, 2),
                    "cs_parts_nonempty_pct": round(cs_rate * 100, 2),
                    "eng_span_precision_vs_anchor_pct": round(p * 100, 2),
                    "eng_span_recall_vs_anchor_pct": round(r * 100, 2),
                    "eng_span_f1_vs_anchor_pct": round(f1 * 100, 2),
                    "composite_score_pct": round(composite * 100, 2),
                }
            )

        score_df = pd.DataFrame(score_rows).sort_values(by="composite_score_pct", ascending=False)
        return df, score_df

    def export(self, df: pd.DataFrame, score_df: pd.DataFrame) -> Path:
        out = Path(self.args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        stats_df = pd.DataFrame(
            [
                {"Metric": "Target Sentences", "Value": self.args.limit},
                {"Metric": "Collected Sentences", "Value": len(df)},
                {"Metric": "Minimum Words", "Value": self.args.min_words},
                {"Metric": "Minimum ENG Words", "Value": self.args.min_eng_words},
                {
                    "Metric": "EST Dominant Required",
                    "Value": "no" if self.args.allow_eng_dominant else "yes",
                },
                {"Metric": "Posts Scraped", "Value": self.stats["posts_scraped"]},
                {"Metric": "Comments Scraped", "Value": self.stats["comments_scraped"]},
                {"Metric": "Sentences Processed", "Value": self.stats["sentences_processed"]},
                {"Metric": "Candidates Kept", "Value": self.stats["candidates_kept"]},
                {"Metric": "Visited Posts", "Value": len(self.visited_posts)},
                {"Metric": "Restarts", "Value": self.stats.get("restarts", 0)},
                {"Metric": "Created At", "Value": self.stats["date"]},
            ]
        )

        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Sentences", index=False)
            score_df.to_excel(writer, sheet_name="IdentifierScores", index=False)
            stats_df.to_excel(writer, sheet_name="RunStats", index=False)

            ws = writer.sheets["Sentences"]
            ws.column_dimensions["A"].width = 6
            ws.column_dimensions["B"].width = 90
            ws.column_dimensions["C"].width = 45
            ws.column_dimensions["D"].width = 12
            ws.freeze_panes = "A2"

            sw = writer.sheets["IdentifierScores"]
            for col in ("A", "B", "C", "D", "E", "F", "G", "H"):
                sw.column_dimensions[col].width = 28

        return out


def main() -> int:
    args = parse_args()
    pipeline = RedditCodeSwitchPipeline(args)
    print("\nCollecting Reddit sentences...")
    completed = pipeline.collect()
    restart_cap = max(0, int(args.max_restarts))
    while (not completed) and (len(pipeline.rows) < args.limit) and (pipeline.stats["restarts"] < restart_cap):
        pipeline.stats["restarts"] += 1
        print(
            f"Restarting collect cycle {pipeline.stats['restarts']}/{restart_cap} "
            f"(current rows: {len(pipeline.rows)})..."
        )
        # Värske seanss aitab pärast ajutist piirangut/võrgu seiskumist
        pipeline.scraper = RedditScraper()
        completed = pipeline.collect()

    print(f"Collected {len(pipeline.rows)} candidate sentences")
    if args.collect_only:
        pipeline._save_checkpoint(force=True)
        print("Collect-only mode complete. Checkpoint saved.")
        return 0

    print("\nRunning identifier analysis...")
    df, score_df = pipeline.annotate()
    out = pipeline.export(df, score_df)
    pipeline._save_checkpoint(force=True)

    print("\nDone.")
    print(f"Output: {out}")
    print(f"Rows: {len(df)}")
    print("\nTop identifier scores:")
    print(score_df.head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

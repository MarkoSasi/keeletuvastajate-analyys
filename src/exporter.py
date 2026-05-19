import pandas as pd
import re
from pathlib import Path
from typing import Optional
from config import OUTPUT_FILE
from utils import count_words, WORD_RE, ESTONIAN_SUFFIXES


class CorpusExporter:
    """Ekspordi korpus Exceli kujul märgendamiseks."""

    # Tavalised ingliskeelsed sõnad
    _ENGLISH_MARKERS = {
        "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "must", "can",
        "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
        "this", "that", "these", "those", "what", "which", "who",
        "at", "in", "on", "of", "for", "to", "with", "by", "from",
        "think", "know", "want", "need", "like", "get", "make", "go",
        "good", "bad", "new", "old", "big", "small", "nice", "cool",
        "really", "very", "just", "also", "too", "now", "then",
        "something", "anything", "nothing", "everything",
        "always", "never", "sometimes", "maybe", "probably",
    }

    def __init__(self, output_path: Optional[str] = None):
        """Initsialiseeri eksportija."""
        self.output_path = Path(output_path or OUTPUT_FILE)

    def _count_words(self, text: str) -> int:
        """Loe sõnade arv tekstis."""
        return count_words(text)

    def _detect_foreign_words(self, text: str) -> dict:
        """
        Tuvasta võimalikud võõrsõnad ja jaota need kategooriatesse.

        Tagastab sõnastiku järgmiste võtmetega:
        - 'pure_foreign': sõnad, mis tunduvad puhtinglise (ilma eesti sufiksita)
        - 'with_suffix': sõnad, mis näivad inglise tüvi + eesti sufiks
        - 'all_foreign': kõik tuvastatud võõrsõnad
        """
        words = re.findall(r'[a-zA-ZäöüõÄÖÜÕšžŠŽ]+', text)

        pure_foreign = []
        with_suffix = []

        for word in words:
            word_lower = word.lower()

            # Jäta väga lühikesed sõnad vahele
            if len(word) < 3:
                continue

            # Kontrolli, kas see on teadaolev ingliskeelne sõna
            if word_lower in self._ENGLISH_MARKERS:
                pure_foreign.append(word)
                continue

            # Kontrolli, kas sõnal on eesti järelliide, aga ingliskeelne tüvi
            for suffix in ESTONIAN_SUFFIXES:
                if word_lower.endswith(suffix) and len(word) > len(suffix) + 2:
                    root = word[:-len(suffix)]
                    # Kui tüvi ei sisalda eesti-spetsiifilisi tähemärke, võib olla inglise+järelliide
                    if not any(c in root.lower() for c in 'äöüõšž'):
                        # Kontrolli, kas tüvi näeb välja ingliskeelne (puudub eesti keelele tüüpiline keeruline kaashäälikuühend)
                        if self._looks_english(root):
                            with_suffix.append(f"{word} ({root}+{suffix})")
                            break

        return {
            'pure_foreign': pure_foreign,
            'with_suffix': with_suffix,
            'all_foreign': pure_foreign + [w.split()[0] for w in with_suffix]
        }

    def _looks_english(self, word: str) -> bool:
        word = word.lower()

        # Tavalised ingliskeelsete sõnade mustrid
        english_patterns = [
            'ing', 'tion', 'ment', 'ness', 'able', 'ible', 'ful', 'less',
            'er', 'or', 'ist', 'ism', 'ity', 'ous', 'ive', 'al', 'ic',
        ]

        # Kontrolli tavalisi inglise lõppe
        for pattern in english_patterns:
            if word.endswith(pattern):
                return True

        # Kontrolli tavalisi inglise tähekombinatsioone
        english_combos = ['th', 'sh', 'ch', 'wh', 'ph', 'ck', 'qu', 'ght']
        for combo in english_combos:
            if combo in word:
                return True

        # Kui sõna on laiemas loendis, on see tõenäoliselt ingliskeelne
        common_roots = {
            'meet', 'work', 'team', 'game', 'play', 'stream', 'post', 'like',
            'share', 'comment', 'follow', 'start', 'train', 'sport', 'fitness',
            'design', 'brand', 'market', 'project', 'manager', 'developer',
            'computer', 'laptop', 'phone', 'screen', 'film', 'music', 'video',
            'cool', 'nice', 'great', 'super', 'random', 'weird', 'crazy',
        }
        
        for root in common_roots:
            if word.startswith(root) or word == root:
                return True
        
        return False
    
    def export_with_stats(
        self,
        sentences: list[dict],
        stats: Optional[dict] = None
    ) -> Path:
        """
        Ekspordi laused koos märgendusveergude ja eelnevalt tuvastatud võõrsõnadega.
        """
        data = []
        for i, sent in enumerate(sentences, 1):
            sentence_text = sent.get("sentence", "")
            detection = self._detect_foreign_words(sentence_text)

            row = {
                "id": i,
                "sentence": sentence_text,
                "word_count": self._count_words(sentence_text),
                "detected_spans": sent.get("detected_spans", ""),
                # Eelnevalt tuvastatud võõrsõnad (käsimärgenduse abiks)
                "detected_foreign": ", ".join(detection['pure_foreign'][:10]),
                "detected_with_suffix": ", ".join(detection['with_suffix'][:5]),
                # Käsimärgenduse veerud
                "type_A_pure_foreign": "",  # Võõrsõnad ilma lõppudeta
                "type_B_foreign_suffix": "",  # Võõrsõna + eesti sufiks
                "type_C_phonetic": "",  # Foneetiline kirjapilt (+ võimalik sufiks)
                "switch_type": "",  # lausesisene, lausetevaheline
                "notes": "",
                # Metaandmed
                "source_url": sent.get("source_url", ""),
                "source_type": sent.get("source_type", ""),
            }
            data.append(row)

        df = pd.DataFrame(data)

        # Koosta statistika DataFrame
        if stats is None:
            stats = {}
        
        stats_data = [
            {"Metric": "Total Sentences", "Value": len(sentences)},
            {"Metric": "Total Words", "Value": df["word_count"].sum()},
            {"Metric": "Average Words per Sentence", "Value": round(df["word_count"].mean(), 2) if len(sentences) > 0 else 0},
            {"Metric": "Posts Scraped", "Value": stats.get("posts_scraped", "N/A")},
            {"Metric": "Comments Scraped", "Value": stats.get("comments_scraped", "N/A")},
            {"Metric": "Keywords Used", "Value": stats.get("keywords_used", "N/A")},
            {"Metric": "Collection Date", "Value": stats.get("date", "N/A")},
        ]
        stats_df = pd.DataFrame(stats_data)

        # Märgendamise juhend
        guide_data = [
            {"Category": "Type A: Pure Foreign", 
             "Description": "Foreign words/phrases without any Estonian inflectional endings",
             "Examples": "nice, cool, meeting, software, random"},
            {"Category": "Type B: Foreign + Suffix", 
             "Description": "Foreign words/phrases with Estonian inflectional ending attached",
             "Examples": "meetingule, computeriga, softwareks, cooliks"},
            {"Category": "Type C: Phonetic", 
             "Description": "Foreign words written phonetically (may also have Estonian endings)",
             "Examples": "miiting, kompuuter, kuul, softveer"},
            {"Category": "Switch Types", 
             "Description": "Types of code-switching",
             "Examples": "intra-sentential (within sentence), inter-sentential (between sentences), tag-switching (fillers/tags)"},
        ]
        guide_df = pd.DataFrame(guide_data)

        # Ekspordi Exceli
        with pd.ExcelWriter(self.output_path, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Corpus')
            stats_df.to_excel(writer, index=False, sheet_name='Stats')
            guide_df.to_excel(writer, index=False, sheet_name='Tagging Guide')

            # Vorminda korpuse leht
            worksheet = writer.sheets['Corpus']
            column_widths = {
                'A': 6,    # id
                'B': 80,   # sentence
                'C': 10,   # word_count
                'D': 60,   # detected_spans
                'E': 40,   # detected_foreign
                'F': 40,   # detected_with_suffix
                'G': 35,   # type_A_pure_foreign
                'H': 35,   # type_B_foreign_suffix
                'I': 35,   # type_C_phonetic
                'J': 18,   # switch_type
                'K': 25,   # notes
                'L': 45,   # source_url
                'M': 12,   # source_type
            }
            for col, width in column_widths.items():
                worksheet.column_dimensions[col].width = width
            worksheet.freeze_panes = 'A2'

            # Vorminda statistika leht
            stats_sheet = writer.sheets['Stats']
            stats_sheet.column_dimensions['A'].width = 30
            stats_sheet.column_dimensions['B'].width = 20

            # Vorminda juhendi leht
            guide_sheet = writer.sheets['Tagging Guide']
            guide_sheet.column_dimensions['A'].width = 25
            guide_sheet.column_dimensions['B'].width = 60
            guide_sheet.column_dimensions['C'].width = 50

        return self.output_path

    def export(
        self,
        sentences: list[dict],
        include_empty_columns: bool = True
    ) -> Path:
        """Põhiline eksport (kutsub export_with_stats ilma statistikaga)."""
        return self.export_with_stats(sentences, None)


if __name__ == "__main__":
    # Testi eksportijat
    test_sentences = [
        {
            "sentence": "Ma arvan et this is really interesting stuff here",
            "source_url": "https://reddit.com/r/Eesti/comments/abc123",
            "source_type": "comment"
        },
        {
            "sentence": "Lähen meetingule koos teamiga, see on super cool projekti jaoks",
            "source_url": "https://reddit.com/r/Eesti/comments/def456",
            "source_type": "post_title"
        },
        {
            "sentence": "Nägin seda streamis ja oli totally random moment",
            "source_url": "https://reddit.com/r/Eesti/comments/ghi789",
            "source_type": "comment"
        },
    ]
    
    exporter = CorpusExporter("test_tagging.xlsx")
    path = exporter.export_with_stats(
        test_sentences,
        {"posts_scraped": 100, "comments_scraped": 500, "date": "2026-01-15"}
    )
    print(f"Exported to: {path}")

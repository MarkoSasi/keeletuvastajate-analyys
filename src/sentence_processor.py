"""
Lausetöötlus: puhastamine, valideerimine ja duplikaatide eemaldamine.
"""

import re
from config import MIN_WORDS
from utils import clean_text, count_words


class SentenceProcessor:

    def __init__(self):
        # URL-i muster
        self.url_pattern = re.compile(
            r'https?://\S+|www\.\S+|[\w\.-]+\.(com|org|net|io|ee|ru|co|uk|de|fi)/\S*',
            re.IGNORECASE
        )

        # Redditi-spetsiifilised puhastamise mustrid
        self.reddit_patterns = [
            re.compile(r'\[deleted\]', re.IGNORECASE),
            re.compile(r'\[removed\]', re.IGNORECASE),
            re.compile(r'u/\w+'),  # u/kasutajanime mainimised
            re.compile(r'r/\w+'),  # r/subreddit mainimised
            re.compile(r'&amp;'),  # HTML-olemid
            re.compile(r'&lt;'),
            re.compile(r'&gt;'),
            re.compile(r'&nbsp;'),
            re.compile(r'\*\*|\*|~~|__'),  # Markdowni vormindus
            re.compile(r'^#+\s*'),  # Markdowni pealkirjad
            re.compile(r'^\s*[-*]\s+'),  # Täppidega loend
            re.compile(r'^\s*\d+\.\s+'),  # Nummerdatud loend
            re.compile(r'\s+'),  # Mitu tühikut üheks
        ]

        # Lauset lõpetav kirjavahemärk
        self.sentence_endings = re.compile(r'[.!?]+')

        # Juba nähtud laused duplikaatide tuvastamiseks
        self.seen_sentences = set()

    def clean_text(self, text: str) -> str:
        """Eemalda tekstist URL-id ja Redditi-spetsiifiline vormindus."""
        return clean_text(text)

    def split_into_sentences(self, text: str) -> list[str]:
        """Jaga tekst üksikuteks lauseteks."""
        if not text:
            return []

        # Puhasta enne jagamist
        text = self.clean_text(text)

        if not text:
            return []

        # Jaga lause lõpu järgi, säilita eraldaja
        parts = self.sentence_endings.split(text)

        sentences = []
        for part in parts:
            part = part.strip()
            if part:
                sentences.append(part)

        # Jaga ka reavahetuste järgi, kui lauselõppu ei leitud
        if len(sentences) <= 1 and '\n' in text:
            sentences = [s.strip() for s in text.split('\n') if s.strip()]

        return sentences

    def count_words(self, text: str) -> int:
        """Loe tegelikud sõnad (mitte kirjavahemärgid ega ainult numbrid)."""
        return count_words(text)

    def is_valid_sentence(self, sentence: str) -> bool:
        """Kontrolli, kas lause vastab miinimumnõuetele."""
        if not sentence:
            return False

        word_count = self.count_words(sentence)

        # Peab olema vähemalt MIN_WORDS sõna
        if word_count < MIN_WORDS:
            return False

        # Pärast puhastamist ei tohi olla vaid URL-id/lingid
        cleaned = self.clean_text(sentence)
        if self.count_words(cleaned) < MIN_WORDS:
            return False

        return True

    def is_duplicate(self, sentence: str) -> bool:
        """Kontrolli, kas seda lauset on juba nähtud."""
        # Normaliseeri võrdluseks
        normalized = self._normalize_for_comparison(sentence)

        if normalized in self.seen_sentences:
            return True

        self.seen_sentences.add(normalized)
        return False

    def _normalize_for_comparison(self, text: str) -> str:
        """Normaliseeri tekst duplikaatide tuvastamiseks."""
        # Väiketähed, eemalda lisatühikud ja kirjavahemärgid
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = ' '.join(text.split())
        return text

    def process_text(self, text: str) -> list[str]:
        """
        Teksti täielik töötlusprotsess.

        Args:
            text: Toortekst Redditist

        Returns:
            Sobivate, puhastatud ja unikaalsete lausete loend
        """
        # Jaga lauseteks
        sentences = self.split_into_sentences(text)

        valid_sentences = []
        for sentence in sentences:
            # Puhasta lause
            cleaned = self.clean_text(sentence)

            # Kontrolli sobivust
            if not self.is_valid_sentence(cleaned):
                continue

            # Kontrolli duplikaate
            if self.is_duplicate(cleaned):
                continue

            valid_sentences.append(cleaned)

        return valid_sentences

    def reset_duplicates(self):
        """Tühjenda nähtud lausete hulk."""
        self.seen_sentences.clear()

    def get_seen_count(self) -> int:
        """Tagasta nähtud unikaalsete lausete arv."""
        return len(self.seen_sentences)


if __name__ == "__main__":
    # Testi protsessorit
    processor = SentenceProcessor()

    test_texts = [
        "Check out https://example.com for more info. See on teine lause which is longer.",
        "This is a short one.",  # Liiga lühike
        "Ma arvan et this whole thing is really interesting and we should discuss it more.",
        "Same sentence again. Ma arvan et this whole thing is really interesting and we should discuss it more.",
    ]
    
    print("Testing sentence processor:\n")
    for text in test_texts:
        results = processor.process_text(text)
        print(f"Input: {text[:60]}...")
        print(f"Output sentences: {results}")
        print()
    
    print(f"Total unique sentences: {processor.get_seen_count()}")

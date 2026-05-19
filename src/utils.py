

import html
import re
from typing import List


# Iga projekti moodul peaks kasutama seda enda regexi asemel.
TOKEN_RE = re.compile(r"[a-zA-ZäöüõÄÖÜÕšžŠŽ']+")

# Range variant – ainult tähed, ilma ülakomata (kasutatakse sõnade lugemiseks).
WORD_RE = re.compile(r"[a-zA-ZäöüõÄÖÜÕšžŠŽ]+")


def tokenize(text: str) -> List[str]:
    """Eralda *text*-ist sõned (koos ülakomadega)."""
    return TOKEN_RE.findall(text)


def count_words(text: str) -> int:
    """Loe ainult tähtedest koosnevad sõnad (ilma ülakomade ja numbriteta)."""
    return len(WORD_RE.findall(text))


ESTONIAN_SUFFIXES = (
    # Käändelõpud
    "sse", "st", "le", "lt", "ga", "ta", "ks", "ni", "na",
    # Tegusõnalõpud
    "ma", "da", "nud", "tud", "dud", "des", "mata",
    # Sagedased tuletuslõpud
    "ist", "iga", "ile", "ilt",
)


def has_estonian_suffix(word: str) -> bool:
    """
    Kontrolli, kas *word* kannab järelliidet.

    Tagastab True ainult siis, kui sõna sisaldab ka eesti keelele
    omaseid tähemärke (äöüõšž), vältides valepositiive inglise
    sõnade puhul.
    """
    word = word.lower()
    if not any(ch in word for ch in "äöüõšž"):
        return False
    for suffix in ESTONIAN_SUFFIXES:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return True
    return False


# ── Teksti puhastamine ────────────────────────────────────────────────────

_URL_RE = re.compile(
    r"https?://\S+|www\.\S+|[\w.\-]+\.(com|org|net|io|ee|ru|co|uk|de|fi)/\S*",
    re.IGNORECASE,
)
_FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`]*`")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((?:https?://|www\.)[^)]+\)")
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_GIF_IMG_RE = re.compile(r"\[(gif|img)\]\([^)]+\)")
_REDDIT_USER_RE = re.compile(r"u/\w+")
_REDDIT_SUB_RE = re.compile(r"r/\w+")
_QUOTE_RE = re.compile(r"^\s*>+\s*", re.MULTILINE)
_MD_FMT_RE = re.compile(r"\*\*|\*|~~|__")
_MD_HEADER_RE = re.compile(r"^#+\s*", re.MULTILINE)
_BULLET_RE = re.compile(r"^\s*[-*]\s+", re.MULTILINE)
_NUMLIST_RE = re.compile(r"^\s*\d+\.\s+", re.MULTILINE)
_LANG_PREFIX_RE = re.compile(r"^\s*[A-Z]{2,5}\s*:\s*")
_MULTI_WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """
    Eemalda *text*-ist markdown, URL-id, Redditi vormindus ja HTML-i olemid.

    See on projekti ainus puhastusfunktsioon.
    """
    if not text:
        return ""

    text = html.unescape(str(text))

    text = _FENCED_CODE_RE.sub(" ", text)
    text = _INLINE_CODE_RE.sub(" ", text)

    # Markdowni lingid – säilita lingi tekst, kustuta URL
    text = _MD_LINK_RE.sub(r"\1", text)

    # Pildid / gif-id
    text = _MD_IMAGE_RE.sub("", text)
    text = _GIF_IMG_RE.sub("", text)

    # URL-id
    text = _URL_RE.sub("", text)

    # Redditi-spetsiifiline
    text = _REDDIT_USER_RE.sub("", text)
    text = _REDDIT_SUB_RE.sub("", text)
    text = _QUOTE_RE.sub("", text)

    # Markdowni vormindus
    text = _MD_FMT_RE.sub(" ", text)
    text = _MD_HEADER_RE.sub(" ", text)
    text = _BULLET_RE.sub(" ", text)
    text = _NUMLIST_RE.sub(" ", text)

    # Keele-eesliited (nt "ENG: ...")
    text = _LANG_PREFIX_RE.sub("", text)

    text = _MULTI_WS_RE.sub(" ", text).strip()

    return text

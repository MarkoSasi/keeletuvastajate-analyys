"""
Eesti-inglise koodivahetuse tuvastamine.
"""

from lingua import Language, LanguageDetectorBuilder
from typing import Dict, List
import re

from utils import clean_text as _utils_clean_text, has_estonian_suffix, WORD_RE


class LanguageDetector:
    """Eesti-inglise koodivahetuse tuvastaja tekstis."""

    def __init__(self):
        # Loo tuvastaja ainult eesti ja inglise keele jaoks
        self.detector = LanguageDetectorBuilder.from_languages(
            Language.ESTONIAN, Language.ENGLISH
        ).with_preloaded_language_models().build()

        # KINDLAD eesti sõnad – need on selgelt eesti, mitte mitmetähenduslikud
        self.estonian_markers_definite = {
            # Asesõnad (ilma 'me'-ta, mis on inglise keeles)
            "ma", "sa", "ta", "te", "nad", "mina", "sina", "tema", "meie", "teie", "nemad",
            # Küsisõnad (ilma 'see', 'need', 'too' – inglise sõnad)
            "nood", "kes", "mis", "kus", "kuidas", "miks", "millal",
            # Tegusõnad – olema (ilma 'on' – inglise eessõna)
            "olen", "oled", "oleme", "olete", "oli", "olid", "olime", "oleks", "olnud",
            # Tegusõnad – saama (ilma 'said' – inglise 'say' minevik)
            "saan", "saad", "saame", "saab", "sai", "saaks", "saanud",
            # Tegusõnad – tegema
            "teen", "teed", "teeme", "teeb", "tegi", "teeks", "teinud", "teha",
            # Tegusõnad – minema/tulema
            "lähen", "läheb", "läks", "tulen", "tuleb", "tuli",
            # Muud sagedased tegusõnad
            "võib", "peab", "tahab", "teab", "oskab", "peaks", "võiks",
            "tahan", "arvan", "arvab", "mõtlen", "usun", "näen", "kuulen",
            # Sidesõnad
            "ja", "ning", "aga", "kuid", "ent", "või", "ehk", "et", "kui", "sest", "kuna",
            "siis", "nii", "ka", "kas", "pole", "ei", "mitte",
            # Sagedased määrsõnad
            "väga", "palju", "rohkem", "veel", "juba", "nüüd", "täna", "homme", "eile",
            "siin", "seal", "kuhu", "kust", "alati", "kunagi", "vahel",
            # Muud sagedased sõnad
            "oma", "mingi", "kõik", "ise", "ainult", "vaid", "ikkagi", "küll",
            "kindlasti", "ilmselt", "tegelikult", "muidugi", "vist",
        }

        # MITMETÄHENDUSLIKUD sõnad – kehtivad ka inglise keeles
        self.estonian_markers_ambiguous = {
            "on",    # Eesti "on" vs inglise eessõna
            "see",   # Eesti "see" vs inglise tegusõna "see"
            "need",  # Eesti "need" vs inglise tegusõna "need"
            "me",    # Eesti "me" vs inglise "me"
            "too",   # Eesti "too" vs inglise "also"
            "said",  # Eesti "said" vs inglise "say" minevik
            "just",  # Eesti "just" vs inglise "just"
            "no",    # Eesti "no" vs inglise "no"
            "a",     # Eesti murde "aga" vs inglise artikkel "a"
        }

        # Mustrid, mida EIRATA – brändinimed ja laensõnad, mis ei viita koodivahetusele
        self.ignore_patterns = {
            # Brändinimed ja pärisnimed
            "paypal", "facebook", "google", "youtube", "reddit", "twitter", "instagram",
            "spotify", "netflix", "amazon", "apple", "microsoft", "tiktok", "snapchat",
            "uber", "bolt", "wolt", "wise", "revolut", "swedi", "swedbank",
            "ielts", "toefl", "covid", "euro", "bitcoin", "crypto",
            "guardian", "times", "post", "bbc", "cnn", "fox", "news",
            # Eesti keelde täielikult integreerunud laensõnad
            "ok", "super", "extra", "mega", "mini", "maxi",
            "laptop", "computer", "internet", "wifi", "bluetooth",
            "app", "link",
            "pizza", "burger", "cocktail", "brunch",
            "fitness", "gym", "sport", "team",
            "design", "style", "brand", "trend", "fashion",
            "test", "stress", "project", "manager", "meeting",
            "marketing", "business", "startup",
            "register", "format", "standard", "system", "process",
        }

        # Tugevad inglise funktsioonisõnad – osutavad tegelikele inglise fraasidele
        self.english_function_words = {
            # Artiklid (tugevaimad osutid)
            "the", "a", "an",
            # Asesõnad
            "i", "you", "he", "she", "it", "we", "they",
            "my", "your", "his", "her", "its", "our", "their",
            "me", "him", "us", "them",
            # Tegusõnad
            "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "can", "may", "might", "must",
            # Eessõnad ja sidesõnad
            "of", "in", "to", "for", "with", "on", "at", "from", "by",
            "about", "into", "through", "during", "before", "after",
            "because", "although", "while", "if", "unless", "until",
            # Muud funktsioonisõnad
            "not", "no", "yes", "very", "really", "just", "also", "too",
            "this", "that", "these", "those", "what", "which", "who", "how", "why",
            "there", "here", "where", "when",
        }

        self.english_content_words = {
            # Sagedased tegusõnad
            "think", "know", "want", "need", "like", "love", "hate",
            "say", "tell", "ask", "answer", "speak", "talk",
            "come", "go", "get", "give", "take", "make", "put", "went", "got",
            "see", "look", "watch", "hear", "listen", "feel", "find", "found",
            "work", "try", "use", "call", "help", "start", "stop", "keep",
            "read", "write", "learn", "teach", "play", "run", "walk", "move",
            "wait", "hope", "believe", "understand", "remember", "forget",
            "show", "mean", "buy", "sell", "pay", "choose", "pick", "leave",
            "bring", "send", "receive", "hold", "break", "fix", "build", "create",
            "check", "test", "review", "compare", "share", "follow", "join",
            "change", "grow", "die", "live", "kill", "save", "spend", "waste",
            "miss", "hit", "win", "lose", "fail", "succeed", "happen", "become",
            # Omadussõnad
            "good", "bad", "new", "old", "big", "small", "long", "short",
            "right", "wrong", "true", "false", "real", "fake",
            "nice", "great", "best", "worst", "better", "worse", "amazing", "awesome",
            "interesting", "boring", "important", "different", "same", "special",
            "easy", "hard", "difficult", "simple", "complex", "complicated",
            "happy", "sad", "angry", "scared", "excited", "tired", "sick",
            "hot", "cold", "warm", "cool", "wet", "dry", "clean", "dirty",
            "free", "busy", "ready", "late", "early", "fast", "slow",
            "full", "empty", "open", "closed", "safe", "dangerous",
            "strong", "weak", "rich", "poor", "young", "cheap", "expensive",
            "crazy", "weird", "strange", "funny", "serious", "stupid", "smart",
            "random", "normal", "perfect", "terrible", "horrible", "beautiful", "ugly",
            # Nimisõnad
            "something", "anything", "nothing", "everything",
            "someone", "anyone", "everyone", "nobody",
            "thing", "things", "person", "people", "place", "time", "way",
            "world", "life", "day", "year", "week", "month",
            "money", "work", "home", "school", "job", "business",
            "weather", "outside", "inside", "morning", "evening", "night",
            "idea", "problem", "question", "answer", "reason", "example",
            "friend", "family", "guy", "girl", "dude", "man", "woman", "kid",
            "stuff", "shit", "fuck", "damn", "hell", "crap",  # Sagedased vandesõnad
            "game", "movie", "music", "book", "story", "news", "show",
            "phone", "car", "house", "room", "door", "window",
            "food", "water", "beer", "coffee", "tea", "wine",
            "price", "cost", "deal", "sale", "discount",
            "point", "part", "side", "end", "beginning", "middle",
            # Määrsõnad
            "always", "never", "sometimes", "often", "usually",
            "now", "then", "soon", "later", "already", "still",
            "today", "yesterday", "tomorrow",
            "actually", "basically", "literally", "probably", "maybe",
            "really", "very", "quite", "pretty", "extremely", "totally",
            "well", "badly", "fast", "slowly", "quickly",
            "anyway", "however", "though", "although", "especially",
            "definitely", "obviously", "apparently", "honestly", "frankly",
            # Sagedased fraasid/täitesõnad
            "okay", "alright", "yeah", "yes", "nope", "sure", "whatever",
            "please", "thanks", "sorry", "excuse",
            # Tehnoloogia sõnad
            "online", "offline", "website", "account", "password", "login",
            "update", "download", "upload", "stream", "post", "comment",
            "message", "email", "call", "text", "chat",
        }
    
    def _clean_text(self, text: str) -> str:
        """Eemalda markdown, URL-id ja muu müra."""
        return _utils_clean_text(text)

    def _tokenize(self, text: str) -> list[str]:
        """Eralda tekstist sõnad."""
        text = self._clean_text(text)
        words = WORD_RE.findall(text)
        return [w for w in words if len(w) >= 2]

    def _classify_word_for_spans(self, word: str) -> str:
        """
        Klassifitseeri üksik sõna märgenduse jaoks.
        Tagastab EST, ENG, AMB või OTHER.
        """
        word_lower = word.lower()

        if word_lower in self.ignore_patterns:
            return "OTHER"

        if self._has_estonian_suffix(word_lower):
            return "EST"

        if word_lower in self.estonian_markers_definite:
            return "EST"

        if word_lower in self.estonian_markers_ambiguous:
            return "AMB"

        if word_lower in self.english_function_words or word_lower in self.english_content_words:
            return "ENG"

        if len(word_lower) < 2:
            return "OTHER"

        lang = self.detector.detect_language_of(word)
        if lang == Language.ESTONIAN:
            return "EST"
        if lang == Language.ENGLISH:
            return "ENG"
        return "OTHER"

    def _resolve_ambiguous_labels(self, labels: List[str]) -> List[str]:
        resolved = labels[:]
        n = len(resolved)

        for i, label in enumerate(resolved):
            if label != "AMB":
                continue

            prev_lang = None
            next_lang = None

            j = i - 1
            while j >= 0:
                if resolved[j] in {"EST", "ENG"}:
                    prev_lang = resolved[j]
                    break
                j -= 1

            j = i + 1
            while j < n:
                if resolved[j] in {"EST", "ENG"}:
                    next_lang = resolved[j]
                    break
                j += 1

            if prev_lang and next_lang and prev_lang == next_lang:
                resolved[i] = prev_lang
            elif prev_lang and next_lang and prev_lang != next_lang:
                resolved[i] = "EST"
            elif prev_lang:
                resolved[i] = prev_lang
            elif next_lang:
                resolved[i] = next_lang
            else:
                # Konservatiivne vaikevalik selle korpuseülesande jaoks.
                resolved[i] = "EST"

        return resolved

    def label_tokens(self, text: str) -> List[Dict[str, object]]:
        """
        Märgenda iga teksti sõne kui EST/ENG/OTHER.
        """
        cleaned = self._clean_text(text)
        token_matches = list(re.finditer(r"[a-zA-ZäöüõÄÖÜÕšžŠŽ']+", cleaned))

        if not token_matches:
            return []

        tokens = []
        labels = []

        # Tahame tuvastada järjestikuste suurtähelistega sõnade juhud
        # (nt "The Guardian") ja märgendada need ENG asemel OTHER-iks.
        for i, m in enumerate(token_matches):
            token = m.group(0)
            label = self._classify_word_for_spans(token)

            is_capitalized = token[0].isupper() if len(token) > 0 else False


            if label == "ENG" and is_capitalized and token.lower() in self.english_function_words:
                prev_cap = (i > 0 and token_matches[i-1].group(0)[0].isupper())
                next_cap = (i < len(token_matches)-1 and token_matches[i+1].group(0)[0].isupper())
                if prev_cap or next_cap:
                    label = "OTHER"

            tokens.append({
                "token": token,
                "start": m.start(),
                "end": m.end(),
                "label": label,
            })
            labels.append(label)

        resolved_labels = self._resolve_ambiguous_labels(labels)
        for token_info, resolved in zip(tokens, resolved_labels):
            token_info["label"] = resolved if resolved != "AMB" else "OTHER"

        # Säilita tuntud eesti keele lausealgustused eesti keeles.
        # See takistab fraasi "See on ..." tõmbamist järgneva inglise konteksti sisse.
        sentence_starter_tokens = [
            ("see", "on"),
            ("see", "oli"),
            ("see", "ei"),
            ("mis", "on"),
            ("mis", "oli"),
            ("kes", "on"),
            ("kes", "oli"),
            ("need", "on"),
            ("need", "olid"),
        ]
        lower_tokens = [t["token"].lower() for t in tokens]
        for pair in sentence_starter_tokens:
            if len(lower_tokens) >= 2 and tuple(lower_tokens[:2]) == pair:
                tokens[0]["label"] = "EST"
                tokens[1]["label"] = "EST"
                break

        return tokens

    def label_sentence_spans(self, text: str, include_other: bool = False) -> List[Dict[str, object]]:

        cleaned = self._clean_text(text)
        tokens = self.label_tokens(cleaned)
        if not tokens:
            return []

        spans = []
        current_label = None
        current_start = None
        current_end = None
        current_word_count = 0

        for token in tokens:
            label = token["label"]
            if label == "OTHER" and not include_other:
                if current_label is not None:
                    spans.append({
                        "text": cleaned[current_start:current_end].strip(),
                        "label": current_label,
                        "start": current_start,
                        "end": current_end,
                        "word_count": current_word_count,
                    })
                    current_label = None
                    current_start = None
                    current_end = None
                    current_word_count = 0
                continue

            if current_label is None:
                current_label = label
                current_start = token["start"]
                current_end = token["end"]
                current_word_count = 1
                continue

            if label == current_label:
                current_end = token["end"]
                current_word_count += 1
            else:
                spans.append({
                    "text": cleaned[current_start:current_end].strip(),
                    "label": current_label,
                    "start": current_start,
                    "end": current_end,
                    "word_count": current_word_count,
                })
                current_label = label
                current_start = token["start"]
                current_end = token["end"]
                current_word_count = 1

        if current_label is not None:
            spans.append({
                "text": cleaned[current_start:current_end].strip(),
                "label": current_label,
                "start": current_start,
                "end": current_end,
                "word_count": current_word_count,
            })

        return spans
    
    def detect_languages_in_text(self, text: str) -> dict:
        result = {
            "estonian": False,
            "english": False,
            "estonian_words": [],
            "english_words": [],
            "english_function_count": 0,
            "english_content_count": 0,
        }
        
        if not text or len(text.strip()) < 3:
            return result
        
        words = self._tokenize(text)
        if len(words) < 2:
            return result
        
        ambiguous_found = []
        has_definite_estonian = False

        text_lower = text.lower().strip()
        estonian_phrase_starters = [
            "see on ", "see oli ", "see ei ",  # see on / oli / ei ole
            "mis on ", "mis oli ",              # mis on / oli
            "kes on ", "kes oli ",              # kes on / oli
            "need on ", "need olid ",           # need on / olid
        ]
        for phrase in estonian_phrase_starters:
            if text_lower.startswith(phrase):
                has_definite_estonian = True
                result["estonian"] = True
                result["estonian_words"].extend(phrase.strip().split())
                break

        for word in words:
            word_lower = word.lower()

            # Jäta ignoreeritavad sõnad vahele (brändid, laensõnad)
            if word_lower in self.ignore_patterns:
                continue

            # Jäta vahele sõnad, mis lõpevad eesti järelliitega (tõenäoliselt eesti + inglise tüvi)
            if self._has_estonian_suffix(word_lower):
                result["estonian"] = True
                result["estonian_words"].append(word)
                has_definite_estonian = True
                continue

            if word_lower in self.estonian_markers_definite:
                result["estonian"] = True
                result["estonian_words"].append(word)
                has_definite_estonian = True
                continue

            # Jälgi mitmetähenduslikke 
            if word_lower in self.estonian_markers_ambiguous:
                ambiguous_found.append(word)
                continue

            if word_lower in self.english_function_words:
                result["english"] = True
                result["english_words"].append(word)
                result["english_function_count"] += 1
                continue

            if word_lower in self.english_content_words:
                result["english"] = True
                result["english_words"].append(word)
                result["english_content_count"] += 1
                continue

            # Pikemate tundmatute sõnade puhul kasuta lingua tuvastajat.
            # AGA jäta vahele pärisnimed (suurtähelised) – need on tõenäoliselt kohanimed.
            if len(word) >= 6:
                # Jäta vahele, kui sõna on suure algustähega (tõenäoliselt pärisnimi/kohanimi)
                is_proper_noun = word[0].isupper()
                if is_proper_noun:
                    continue  # Ära loe kohanimesid eesti sõnadeks

                lang = self.detector.detect_language_of(word)
                if lang == Language.ESTONIAN:
                    # Aktsepteeri kui: sõnas on eesti tähemärgid JA see on väga pikk (liitsõna)
                    has_estonian_chars = any(c in word.lower() for c in 'äöüõšž')
                    is_long_word = len(word) >= 12  # Väga pikad sõnad on tõenäoliselt eesti liitsõnad
                    if has_estonian_chars and is_long_word:
                        result["estonian"] = True
                        result["estonian_words"].append(word)
                        has_definite_estonian = True
                # Ära lisa inglise sõnu automaatselt – liiga palju valepositiive

        if has_definite_estonian and ambiguous_found:
            for word in ambiguous_found:
                result["estonian_words"].append(word)

        return result

    def _has_estonian_suffix(self, word: str) -> bool:
        """Kontrolli, kas sõnal on eesti grammatiline järelliide."""
        return has_estonian_suffix(word)

    def is_codeswitched(self, text: str, min_english_words: int = 2) -> bool:
        """
        Kontrolli, kas tekstis esineb eesti-inglise koodivahetust.

        Lõikupõhised kriteeriumid:
        - Tekstis peavad olema nii EST kui ka ENG lõigud.
        - Tekstis peab olema vähemalt `min_english_words` ingliskeelset sõna.
        - Eesti keele konteksti peab olema piisavalt (tavaliselt >=2 EST sõna).
        - Ühesõnalised eesti sissejuhatused on lubatud mustrites nagu
          "Teadusartiklist: ...".
        - Vältimaks valepositiive peab esinema vähemalt üks inglise funktsioonisõna.
        """
        spans = self.label_sentence_spans(text, include_other=False)
        if not spans:
            return False

        estonian_words = sum(s["word_count"] for s in spans if s["label"] == "EST")
        english_words = sum(s["word_count"] for s in spans if s["label"] == "ENG")
        if english_words < min_english_words:
            return False

        cleaned = self._clean_text(text)
        est_spans = [s for s in spans if s["label"] == "EST"]
        allow_single_est_leadin = False
        if estonian_words < 2:
            # Luba ühesõnalised eesti sissejuhatused nagu:
            # "Teadusartiklist: This artefact has ..."
            allow_single_est_leadin = (
                estonian_words == 1
                and len(est_spans) == 1
                and spans[0]["label"] == "EST"
                and spans[0]["start"] == 0
                and cleaned[spans[0]["end"]:].lstrip().startswith(":")
            )
            if not allow_single_est_leadin:
                return False

        # Nõua vähemalt ühte tugevat eesti sõne.
        # See väldib selliste inglise lausete koodivahetuseks lugemist,
        # mis mainivad ainult sõnu nagu "Estonian"/"Estonia".
        est_anchor_count = 0
        for t in self.label_tokens(text):
            if t["label"] != "EST":
                continue
            token_lower = str(t["token"]).lower()
            if (
                token_lower in self.estonian_markers_definite
                or token_lower in self.estonian_markers_ambiguous
                or self._has_estonian_suffix(token_lower)
                or any(ch in token_lower for ch in "äöüõšž")
            ):
                est_anchor_count += 1
        if est_anchor_count < 1 and not allow_single_est_leadin:
            return False

        tokens = self.label_tokens(text)
        english_function_count = sum(
            1
            for t in tokens
            if t["label"] == "ENG" and str(t["token"]).lower() in self.english_function_words
        )
        if english_function_count == 0:
            return False

        labels_present = {s["label"] for s in spans}
        return "EST" in labels_present and "ENG" in labels_present


if __name__ == "__main__":
    detector = LanguageDetector()
    
    test_cases = [
        # Peaks olema FALSE – eesti keel koos laensõnade/brändidega
        ("FALSE", "Iseasi kas tahta näidata seda swedi kontol :D"),
        ("FALSE", "On jah mingi nuts kogunenud PayPal kontole ja ei tahaks vahendust maksta"),
        ("FALSE", "Ma ise tegin IELTS general training taseme testi"),
        ("FALSE", "Mulle õpetati kunagi koolis, et bread on leib"),
        ("FALSE", "Saime omaenda kodukootud sovereign citizeni"),
        ("FALSE", "No minu arust see pole koduveini ja õlle puhul kunagi probleemiks olnud"),
        ("FALSE", "A vaata sellega see probleem et siis nad ei teeni raha"),
        ("FALSE", "Ma tsiteerisin otse The Guardiani artiklist"),
        # Peaks olema TRUE – tegelik koodivahetus
        ("TRUE", "See on really interesting, I think we should discuss it more"),
        ("TRUE", "Ma arvan et this is completely wrong and we need to fix it"),
        ("TRUE", "Täna oli weather outside so nice that I went for a walk"),
        ("TRUE", "Teadusartiklist: This artefact has no known analogues among the hunter-gatherer art"),
    ]

    print("Testing STRICT code-switching detection:\n")
    for expected, sent in test_cases:
        is_cs = detector.is_codeswitched(sent)
        actual = "TRUE" if is_cs else "FALSE"
        match = "✓" if expected == actual else "✗"
        det = detector.detect_languages_in_text(sent)
        print(f"{match} Expected {expected}, Got {actual}")
        print(f"   Text: {sent[:60]}...")
        print(f"   EN words: {det['english_words'][:5]} (func:{det['english_function_count']})")
        print()

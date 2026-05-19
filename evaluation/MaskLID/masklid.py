import fasttext
import numpy as np
import re
import string
from copy import deepcopy

class MaskLID:
    """Klass koodivahetuse keele tuvastamiseks itereeriva maskeerimise abil."""

    def __init__(self, model_path, languages=-1):
        """Initsialiseeri MaskLID klass.

        Args:
            model_path (str): fastTexti mudeli tee.
            languages (int või list, valikuline): Arvestatavate keelte indeksid või märgendite loend. Vaikimisi -1.
        """
        self.model = fasttext.load_model(model_path)
        self.all_labels = []
        if hasattr(self.model, "get_labels"):
            try:
                self.all_labels = self.model.get_labels()
            except Exception:
                self.all_labels = []

        if languages != -1 and isinstance(languages, list):
            if self.all_labels:
                self.labels = [l for l in languages if l in self.all_labels]
            else:
                self.labels = list(dict.fromkeys(languages))
        else:
            self.labels = self.all_labels[:] if self.all_labels else []

        self.language_indices = self._compute_language_indices(self.labels)
        self.output_matrix = None
        try:
            # Mõned fasttexti ehitused pakuvad get_output_matrix, teised mitte.
            self.output_matrix = self.model.get_output_matrix()
        except Exception:
            self.output_matrix = None

    def _compute_language_indices(self, selected_labels):
        """Arvuta valitud keelte indeksid.

        Args:
            languages (int või list): Keelte indeksid või märgendite loend.

        Returns:
            list: Valitud keelte indeksid.
        """
        if not self.all_labels:
            return []
        if selected_labels and isinstance(selected_labels, list):
            return [self.all_labels.index(l) for l in selected_labels if l in self.all_labels]
        return list(range(len(self.all_labels)))

    @staticmethod
    def _split_words(text):
        return [w for w in re.findall(r"[^\s]+", text) if w and w != "</s>"]

    def _softmax(self, x):
        """Arvuta softmax-i väärtused massiivi x iga skoori jaoks.

        Args:
            x (numpy.ndarray): Sisendmassiiv.

        Returns:
            numpy.ndarray: Softmax-i väljund.
        """
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def _normalize_text(self, text):
        """Normaliseeri sisendtekst.

        Args:
            text (str): Sisendtekst.

        Returns:
            str: Normaliseeritud tekst.
        """
        replace_by = " "
        replacement_map = {ord(c): replace_by for c in '\n_:' + '•#{|}' + string.digits}
        text = text.translate(replacement_map)
        return re.sub(r'\s+', ' ', text).strip()

    def predict(self, text, k=1):
        """Ennusta sisendteksti keel.

        Args:
            text (str): Sisendtekst.
            k (int, valikuline): Tagastatavate parimate ennustuste arv. Vaikimisi 1.

        Returns:
            tuple: Parimad ennustatud märgendid ja nende tõenäosused.
        """
        if self.output_matrix is not None:
            sentence_vector = self.model.get_sentence_vector(text)
            result_vector = np.dot(self.output_matrix, sentence_vector)
            if self.language_indices:
                softmax_result = self._softmax(result_vector)[self.language_indices]
                active_labels = self.labels
            else:
                softmax_result = self._softmax(result_vector)
                active_labels = self.all_labels if self.all_labels else self.labels
            top_k_indices = np.argsort(softmax_result)[-k:][::-1]
            top_k_labels = [active_labels[i] for i in top_k_indices]
            top_k_probs = softmax_result[top_k_indices]
            return tuple(top_k_labels), top_k_probs

        # Varuvariant ilma maatriksi API-deta ehituste jaoks: kasuta natiivset fastTexti predict-i.
        # Küsi rohkem märgendeid, seejärel filtreeri valitud keelte kogumi järgi.
        raw_labels, raw_probs = self.model.predict(text, k=max(k, 10))
        filtered = [(lab, prob) for lab, prob in zip(raw_labels, raw_probs) if (not self.labels or lab in self.labels)]
        if not filtered:
            filtered = list(zip(raw_labels, raw_probs))
        top = filtered[:k]
        return tuple([lab for lab, _ in top]), np.array([prob for _, prob in top])

    def compute_v(self, sentence_vector):
        """Arvuta antud lausevektori keelevektorid.

        Args:
            sentence_vector (numpy.ndarray): Lausevektor.

        Returns:
            list: Sorteeritud loend märgenditest ja nendega seotud vektoritest.
        """
        if self.output_matrix is None:
            return []
        idx = self.language_indices if self.language_indices else list(range(len(self.all_labels)))
        labels = self.labels if self.language_indices else (self.all_labels if self.all_labels else self.labels)
        result_vector = np.dot(self.output_matrix[idx, :], sentence_vector)
        return sorted(zip(labels, result_vector), key=lambda x: x[1], reverse=True)

    def compute_v_per_word(self, text):
        """Arvuta keelevektorid sisendteksti iga sõna kohta.

        Args:
            text (str): Sisendtekst.

        Returns:
            dict: Sõnastik iga sõna keelevektoritega.
        """
        text = self._normalize_text(text)
        if hasattr(self.model, "get_line"):
            try:
                words = self.model.get_line(text)[0]
                words = [w for w in words if w not in ['</s>', '</s>']]
            except Exception:
                words = self._split_words(text)
        else:
            words = self._split_words(text)

        # Maatriksipõhine.
        if self.output_matrix is not None:
            subword_ids = [self.model.get_subwords(sw)[1] for sw in words]
            sentence_vector = [np.sum([self.model.get_input_vector(id) for id in sid], axis=0) for sid in subword_ids]

            dict_text = {}
            for i, word in enumerate(words):
                key = f"{i}_{word}"
                dict_text[key] = {'logits': self.compute_v(sentence_vector[i])}
            return dict_text

        # Varuvariant
        dict_text = {}
        for i, word in enumerate(words):
            labels, probs = self.predict(word, k=max(1, len(self.labels)))
            logits = list(zip(labels, probs))
            key = f"{i}_{word}"
            dict_text[key] = {'logits': logits}
        return dict_text

    def mask_label_top_k(self, dict_text, label, top_keep, top_remove):

        dict_remained = deepcopy(dict_text)
        dict_deleted = {}

        for key, value in dict_text.items():
            logits = value['logits']
            labels = [t[0] for t in logits]

            if label in labels[:top_keep]:
                dict_deleted[key] = dict_remained[key]

            if label in labels[:top_remove]:
                dict_remained.pop(key, None)

        return dict_remained, dict_deleted

    @staticmethod
    def get_sizeof(text):
        """Arvuta teksti suurus baitides.

        Args:
            text (str): Sisendtekst.

        Returns:
            int: Teksti suurus baitides.
        """
        return len(text.encode('utf-8'))

    @staticmethod
    def custom_sort(word):
        """Kohandatud sortimisfunktsioon sõnadele.

        Args:
            word (str): Sisendsõna.

        Returns:
            int või float: Sortimisväärtus.
        """
        match = re.match(r'^(\d+)_', word)
        if match:
            return int(match.group(1))
        else:
            return float('inf')  # Tagasta lõpmatus sõnade jaoks, mis ei alga numbritega

    def sum_logits(self, dict_data, label):
        """Arvuta antud märgendi summa.

        Args:
            dict_data (dict): Sõnastik iga sõna keelevektoritega.
            label (str): Märgend, mille summeerida.

        Returns:
            float: Antud märgendi kogusumma.
        """
        total = 0
        for value in dict_data.values():
            logits = value['logits']
            labels = [t[0] for t in logits]
            if label in labels:
                total += logits[labels.index(label)][1]
        return total

    def predict_codeswitch(self, text, beta, alpha, min_prob, min_length, max_lambda=1, max_retry=3, alpha_step_increase=5, beta_step_increase=5):

        # Kiire varuvariant ilma maatriksi API-deta ehituste jaoks.
        if self.output_matrix is None:
            text = self._normalize_text(text)
            if hasattr(self.model, "get_line"):
                try:
                    words = self.model.get_line(text)[0]
                    words = [w for w in words if w not in ['</s>', '</s>']]
                except Exception:
                    words = self._split_words(text)
            else:
                words = self._split_words(text)
            by_label = {}
            for i, w in enumerate(words):
                labels, probs = self.predict(w, k=1)
                if not labels:
                    continue
                top = labels[0]
                by_label.setdefault(top, []).append(f"{i}_{w}")
            out = {}
            for key, vals in by_label.items():
                out[key] = ' '.join([x.split('_', 1)[1] for x in sorted(set(vals), key=self.custom_sort)])
            return out

        info = {}
        index = 0
        retry = 0

        # arvuta v
        dict_data = self.compute_v_per_word(text)

        while index < max_lambda and retry < max_retry:

            # ennusta tekstile
            pred = self.predict(text, k=1)
            label = pred[0][0]

            # salvesta praegune tekst, juhuks kui peab tagasi astuma
            prev_text = text
            dict_data, dict_masked = self.mask_label_top_k(dict_data, label, beta, alpha)

            masked_text = ' '.join(x.split('_', 1)[1] for x in dict_masked.keys())
            text = ' '.join(x.split('_', 1)[1] for x in dict_data.keys())

            # salvesta info
            if self.get_sizeof(masked_text) > min_length or index == 0:
                temp_pred = self.predict(masked_text)

                if (temp_pred[1][0] > min_prob and temp_pred[0][0] == label) or index == 0:
                    info[index] = {
                        'label': label,
                        'text': masked_text,
                        'text_keys': dict_masked.keys(),
                        'size': self.get_sizeof(masked_text),
                        'sum_logit': self.sum_logits(dict_masked, label)
                    }
                    index += 1
                else:
                    text = prev_text
                    beta += beta_step_increase
                    alpha += alpha_step_increase
                    retry += 1
            else:
                text = prev_text
                beta += beta_step_increase
                alpha += alpha_step_increase
                retry += 1

            if self.get_sizeof(text) < min_length:
                break

        
        # järeltöötlus
        post_info = {}
        for value in info.values():
            key = value['label']
            if key in post_info:
                post_info[key].extend(value['text_keys'])
            else:
                post_info[key] = list(value['text_keys'])

        for key in post_info:
            post_info[key] = ' '.join([x.split('_', 1)[1] for x in sorted(set(post_info[key]), key=self.custom_sort)])
                
                
        return post_info

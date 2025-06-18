from lettucedetect.detectors.base import BaseDetector
from lettucedetect.detectors.prompt_utils import Lang
import fasttext
import fasttext.util
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import os

import stanza

ENTITY_SIMILARITY_THRESHOLD = 0.75

class EntitySimilarityDetector(BaseDetector):
    def __init__(self, lang: Lang = "en", word_vector_path: str = None):
        if lang != "en":
            raise NotImplementedError("Only English is currently supported.")

        self.lang = lang

        # Stanza setup (cached, avoids re-downloads)
        if not os.path.exists(os.path.expanduser(f'~/.stanza_resources/{lang}')):
            stanza.download(lang, verbose=False)
        self.nlp = stanza.Pipeline(lang=lang, processors='tokenize,ner', tokenize_no_ssplit=True, verbose=False)

        # FastText vector loading
        if word_vector_path:
            self.word_vectors = fasttext.load_model(word_vector_path)
        else:
            if not os.path.exists("cc.en.300.bin"):
                fasttext.util.download_model('en', if_exists='ignore')
            self.word_vectors = fasttext.load_model("cc.en.300.bin")

        self.get_vector = self.word_vectors.get_word_vector  

    def _get_entity_embeddings(self, entities: list[str]) -> np.ndarray:
        # Return a 2D numpy array: [n_entities, 300]
        embeddings = []
        for ent in entities:
            tokens = ent.split()
            if not tokens:
                continue
            token_vecs = [self.get_vector(t) for t in tokens]
            avg_vec = np.mean(token_vecs, axis=0)
            embeddings.append(avg_vec)
        return np.array(embeddings)

    def _extract_entities(self, text: str) -> list[str]:
        doc = self.nlp(text)
        return list({ent.text for ent in doc.ents})  

    def _predict(self, context: list[str], answer: str, output_format: str = "spans") -> list:
        full_context = " ".join(context)

        # Extract & embed entities
        context_ents = self._extract_entities(full_context)
        answer_ents = self._extract_entities(answer)

        if not context_ents or not answer_ents:
            return []

        context_vecs = self._get_entity_embeddings(context_ents)
        answer_vecs = self._get_entity_embeddings(answer_ents)

        # Compute cosine similarity in bulk
        sims = cosine_similarity(answer_vecs, context_vecs)  # shape: [n_answer, n_context]
        max_sims = np.max(sims, axis=1)

        hallucinated = []
        for i, sim in enumerate(max_sims):
            if sim < ENTITY_SIMILARITY_THRESHOLD:
                ent_text = answer_ents[i]
                if output_format == "tokens":
                    hallucinated.append({
                        "token": ent_text,
                        "pred": 1,
                        "prob": 1.0 - sim
                    })
                elif output_format == "spans":
                    match = re.search(re.escape(ent_text), answer)
                    if match:
                        hallucinated.append({
                            "text": ent_text,
                            "start": match.start(),
                            "end": match.end(),
                            "confidence": 1.0 - sim
                        })
        return hallucinated

    def predict_prompt(self, prompt: str, answer: str, output_format: str = "tokens") -> list:
        return self._predict([prompt], answer, output_format)

    def predict(self, context: list[str], answer: str, question: str | None = None, output_format: str = "tokens") -> list:
        return self._predict(context, answer, output_format)

    def predict_prompt_batch(self, prompts, answers, output_format="tokens") -> list:
        return [self._predict(p, a, output_format) for p, a in zip(prompts, answers)]

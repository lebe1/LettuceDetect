from lettucedetect.detectors.base import BaseDetector
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import Counter
import nltk
from lettucedetect.detectors.prompt_utils import Lang
import re

ROUGE_PRECISION_THRESHOLD = 0.3

class RougeLemmaBasedDetector(BaseDetector):
    def __init__(
        self,
        lang: Lang = "en",
    ):
        """
        Initialize the RougeBasedDetector.

        :param lang: The language of the model.
        """
        if lang != "en":
            raise NotImplementedError("Currently only English ('en') is supported for ROUGE-based detection.")
        
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('omw-1.4')

        self.lang = lang
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def _preprocess(self, text: str) -> list[str]:
        """Lowercase, tokenize, remove stopwords, and apply lemmatization."""
        tokens = word_tokenize(text.lower())

        return [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token.isalnum() and token not in self.stop_words
        ]
    
    def _rouge_1_precision_recall(self, response_tokens: list[str], context_tokens: list[str]) -> tuple[float, float]:
        """Compute ROUGE-1 precision and recall."""
        ref_counts = Counter(context_tokens)
        hyp_counts = Counter(response_tokens)

        overlap = sum(min(hyp_counts[w], ref_counts[w]) for w in hyp_counts)

        precision = overlap / sum(hyp_counts.values()) if hyp_counts else 0.0
        recall = overlap / sum(ref_counts.values()) if ref_counts else 0.0

        return precision, recall

    def _predict(self, context: list[str], answer: str, output_format: str = "spans") -> list:
        """
        Perform hallucination detection using ROUGE-1 based precision and recall.

        :param context: A list of context strings (representing input arguments).
        :param answer: The generated answer to evaluate.
        :param output_format: Either 'spans' (sentence-level) or 'tokens' (word-level).
        :return: A list of hallucinated spans or tokens with confidence scores.
        """
        context_tokens_all = [self._preprocess(c) for c in context]
        # answer_tokens = self._preprocess(answer)

        # print("CONTEXT TOKENS", context_tokens_all)
        # print("ANSWER TOKEN", answer_tokens)

        if output_format == "spans":
            spans = []
            for match in re.finditer(r'[^.?!]+[.?!]', answer):
                sentence = match.group().strip()
                sentence_tokens = self._preprocess(sentence)
                prec, _ = self._rouge_1_precision_recall(sentence_tokens, sum(context_tokens_all, []))
                # print("Sentence tokens", sentence_tokens)
                # print("Precision", prec)

                if prec < ROUGE_PRECISION_THRESHOLD:  # Threshold for hallucination
                    spans.append({
                        "text": sentence,
                        "start": match.start(),
                        "end": match.end(),
                        "confidence": 1 - prec  # Lower precision means higher confidence it's a hallucination
                    })
            return spans

        elif output_format == "tokens":
            token_outputs = []
            for match in re.finditer(r'\b\w+\b', answer):
                token = match.group()
                token_proc = self._preprocess(token)
                if not token_proc:
                    continue
                token_in_context = token_proc[0] in sum(context_tokens_all, [])

                token_outputs.append({
                    "token": token,
                    "pred": int(not token_in_context),
                    "prob": 0.99 if not token_in_context else 0.01
                })
            return token_outputs

        else:
            raise ValueError("Invalid output_format. Use 'tokens' or 'spans'.")
        
    def predict_prompt(self, prompt: str, answer: str, output_format: str = "tokens") -> list:
        """Predict hallucination spans from the provided prompt and answer.

        :param prompt: The prompt string.
        :param answer: The answer string.
        :param output_format: "spans" to return grouped spans.
        """
        return self._predict([prompt], answer, output_format)
    
    def predict(
        self,
        context: list[str],
        answer: str,
        question: str | None = None,
        output_format: str = "tokens",
    ) -> list:
        """Predict hallucination tokens or spans from the provided context, answer, and question.
        This is a useful interface when we don't want to predict a specific prompt, but rather we have a list of contexts, answers, and questions. Useful to interface with RAG systems.

        :param context: A list of context strings.
        :param answer: The answer string.
        :param question: The question string.
        :param output_format: "tokens" to return token-level predictions, or "spans" to return grouped spans.
        """

        print("Entered predict method in RougeBased class")
        return self._predict(context, answer, output_format)
    
    def predict_prompt_batch(self, prompts, answers, output_format="tokens") -> list:
        """Predict hallucination tokens or spans from the provided prompts and answers.

        :param prompts: List of prompt strings.
        :param answers: List of answer strings.
        :param output_format: "tokens" to return token-level predictions, or "spans" to return grouped spans.
        """
        return [self._predict(p, a, output_format) for p, a in zip(prompts, answers)]
    



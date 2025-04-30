from abc import ABC, abstractmethod

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from lettucedetect.datasets.hallucination_dataset import HallucinationDataset

import re
from difflib import SequenceMatcher
import Levenshtein

PROMPT_QA = """
Briefly answer the following question:
{question}
Bear in mind that your response should be strictly based on the following {num_passages} passages:
{context}
In case the passages do not contain the necessary information to answer the question, please reply with: "Unable to answer based on given passages."
output:
"""

PROMPT_SUMMARY = """
Summarize the following text:
{text}
output:
"""

SEQUENCE_MATCH_THRESHOLD = 0.8


class BaseDetector(ABC):
    @abstractmethod
    def _predict(self, context: str, answer: str, output_format: str = "tokens") -> list:
        """Core prediction method to be implemented by subclasses."""
        pass

    def _form_prompt(self, context: list[str], question: str | None) -> str:
        """Form a prompt from the provided context and question. We use different prompts for summary and QA tasks.

        :param context: A list of context strings.
        :param question: The question string.
        :return: The formatted prompt.
        """
        context_str = "\n".join(
            [f"passage {i + 1}: {passage}" for i, passage in enumerate(context)]
        )
        if question is None:
            return PROMPT_SUMMARY.format(text=context_str)
        else:
            return PROMPT_QA.format(
                question=question, num_passages=len(context), context=context_str
            )

    def predict_prompt(self, prompt: str, answer: str, output_format: str = "tokens") -> list:
        """Predict hallucination tokens or spans from the provided prompt and answer.

        :param prompt: The prompt string.
        :param answer: The answer string.
        :param output_format: "tokens" to return token-level predictions, or "spans" to return grouped spans.
        """
        return self._predict(prompt, answer, output_format)

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

        print("Entered predict method in Transformer class")

        prompt = self._form_prompt(context, question)
        return self._predict(prompt, answer, output_format)



class TransformerDetector(BaseDetector):
    def __init__(self, model_path: str, max_length: int = 4096, device=None, **kwargs):
        """Initialize the TransformerDetector.

        :param model_path: The path to the model.
        :param max_length: The maximum length of the input sequence.
        :param device: The device to run the model on.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path, **kwargs)
        self.max_length = max_length
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()



    def _predict(self, context: str, answer: str, output_format: str = "tokens") -> list:
        """Predict hallucination tokens or spans from the provided context and answer.

        :param context: The context string.
        :param answer: The answer string.
        :param output_format: "tokens" to return token-level predictions, or "spans" to return grouped spans.
        """
        # Use the shared tokenization logic from RagTruthDataset
        encoding, labels, offsets, answer_start_token = (
            HallucinationDataset.prepare_tokenized_input(
                self.tokenizer, context, answer, self.max_length
            )
        )

        # Create a label tensor: mark tokens before answer as -100 (ignored) and answer tokens as 0.
        labels = torch.full_like(encoding.input_ids[0], -100, device=self.device)
        labels[answer_start_token:] = 0
        # Move encoding to the device
        encoding = {
            key: value.to(self.device)
            for key, value in encoding.items()
            if key in ["input_ids", "attention_mask", "labels"]
        }
        labels = torch.tensor(labels, device=self.device)

        # Run model inference
        with torch.no_grad():
            outputs = self.model(**encoding)
        logits = outputs.logits
        token_preds = torch.argmax(logits, dim=-1)[0]
        probabilities = torch.softmax(logits, dim=-1)[0]

        # Mask out predictions for context tokens.
        token_preds = torch.where(labels == -100, labels, token_preds)

        if output_format == "tokens":
            # return token probabilities for each token (with the tokens as well, if not -100)
            token_probs = []
            input_ids = encoding["input_ids"][0]  # Get the input_ids tensor from the encoding dict
            for i, (token, pred, prob) in enumerate(zip(input_ids, token_preds, probabilities)):
                if not labels[i].item() == -100:
                    token_probs.append(
                        {
                            "token": self.tokenizer.decode([token]),
                            "pred": pred.item(),
                            "prob": prob[1].item(),  # Get probability for class 1 (hallucination)
                        }
                    )
            return token_probs
        elif output_format == "spans":
            # Compute the answer's character offset (the first token of the answer).
            if answer_start_token < offsets.size(0):
                answer_char_offset = offsets[answer_start_token][0].item()
            else:
                answer_char_offset = 0

            spans: list[dict] = []
            current_span: dict | None = None

            # Iterate over tokens in the answer region.
            for i in range(answer_start_token, token_preds.size(0)):
                # Skip tokens marked as ignored.
                if labels[i].item() == -100:
                    continue

                token_start, token_end = offsets[i].tolist()
                # Skip special tokens with zero length.
                if token_start == token_end:
                    continue

                # Adjust offsets relative to the answer text.
                rel_start = token_start - answer_char_offset
                rel_end = token_end - answer_char_offset

                is_hallucination = (
                    token_preds[i].item() == 1
                )  # assuming class 1 indicates hallucination.
                confidence = probabilities[i, 1].item() if is_hallucination else 0.0

                if is_hallucination:
                    if current_span is None:
                        current_span = {
                            "start": rel_start,
                            "end": rel_end,
                            "confidence": confidence,
                        }
                    else:
                        # Extend the current span.
                        current_span["end"] = rel_end
                        current_span["confidence"] = max(current_span["confidence"], confidence)
                else:
                    # If we were building a hallucination span, finalize it.
                    if current_span is not None:
                        # Extract the hallucinated text from the answer.
                        span_text = answer[current_span["start"] : current_span["end"]]
                        current_span["text"] = span_text
                        spans.append(current_span)
                        current_span = None

            # Append any span still in progress.
            if current_span is not None:
                span_text = answer[current_span["start"] : current_span["end"]]
                current_span["text"] = span_text
                spans.append(current_span)

            return spans
        else:
            raise ValueError("Invalid output_format. Use 'tokens' or 'spans'.")
    

class RuleBasedDetector(BaseDetector):
    def __init__(self):
        pass

    def _predict(self, context: list[str], answer: str, output_format: str = "spans") -> list:
        """
        Perform rule-based hallucination detection on the answer, comparing it to the provided context.

        :param context: A list of context strings to compare against.
        :param answer: The generated answer string to evaluate.
        :param output_format: Output format - either 'spans' (sentence-level) or 'tokens' (word-level).
        :return: A list of hallucination spans or tokens, each with confidence scores.
        """
        # Normalize and join context into a single lowercase string for easier comparison.
        context_str = " ".join(context).lower()

        if output_format == "spans":
            spans = []
            # Iterate over all sentence-like segments in the answer.
            for match in re.finditer(r'[^.?!]+[.?!]', answer):
                sentence = match.group().strip()
                sentence_lower = sentence.lower()

                # Check for hallucination using exact and fuzzy match heuristics.
                if sentence_lower not in context_str and self._fuzzy_levenshtein(sentence_lower, context_str) < SEQUENCE_MATCH_THRESHOLD:
                    spans.append({
                        "text": sentence,
                        "start": match.start(),
                        "end": match.end(),
                        "confidence": 0.99
                    })
            return spans

        elif output_format == "tokens":
            token_outputs = []
            # Check each word-level token in the answer.
            for match in re.finditer(r'\b\w+\b', answer):
                token = match.group()
                token_lower = token.lower()
                is_hallucinated = token_lower not in context_str and self._fuzzy_levenshtein(token_lower, context_str) < SEQUENCE_MATCH_THRESHOLD

                token_outputs.append({
                    "token": token,
                    "pred": int(is_hallucinated),
                    "prob": 0.99 if is_hallucinated else 0.01
                })
            return token_outputs

        else:
            raise ValueError("Invalid output_format. Use 'tokens' or 'spans'.")

    def _fuzzy_sequence_matcher(self, s1: str, s2: str) -> float:
        """
        Compute a fuzzy similarity ratio between two strings using SequenceMatcher.

        :param s1: First string (usually the sentence or token).
        :param s2: Second string (typically the full context).
        :return: A float similarity ratio between 0.0 and 1.0.
        """
        return SequenceMatcher(None, s1, s2).ratio()
    
    def _fuzzy_levenshtein(self, s1: str, s2: str) -> float:
        """
        Compute a fuzzy similarity ratio between two strings using Levenshtein.

        :param s1: First string (usually the sentence or token).
        :param s2: Second string (typically the full context).
        :return: A float similarity ratio between 0.0 and 1.0.
        """
        print("Levenshtein entered")
        return  Levenshtein.ratio(s1, s2)



class HallucinationDetector:
    def __init__(self, method: str = "transformer", **kwargs):
        """Facade for the hallucination detector.

        :param method: "transformer" for the model-based approach, "rule" for the rule-based approach
        :param kwargs: Additional keyword arguments passed to the underlying detector.
        """
        if method == "transformer":
            self.detector = TransformerDetector(**kwargs)
        elif method == "rule":
            self.detector = RuleBasedDetector()
        else:
            raise ValueError("Unsupported method. Choose 'transformer' or 'rule'.")

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
        """
        print("PREDICT method Hallucination class")
        return self.detector.predict(context, answer, question, output_format)

    def predict_prompt(self, prompt: str, answer: str, output_format: str = "tokens") -> list:
        """Predict hallucination tokens or spans from the provided prompt and answer.

        :param prompt: The prompt string.
        :param answer: The answer string.
        """
        return self.detector.predict_prompt(prompt, answer, output_format)

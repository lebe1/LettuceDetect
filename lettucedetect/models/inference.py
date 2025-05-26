import hashlib
import json
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from string import Template
from typing import Literal

import torch
from openai import OpenAI
from transformers import AutoModelForTokenClassification, AutoTokenizer

from lettucedetect.datasets.hallucination_dataset import (
    HallucinationDataset,
)

import re
from difflib import SequenceMatcher
import Levenshtein
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from collections import Counter
import nltk


LANG_TO_PASSAGE = {
    "en": "passage",
    "de": "Passage",
    "fr": "passage",
    "es": "pasaje",
    "it": "brano",
    "pl": "fragment",
}

SEQUENCE_MATCH_THRESHOLD = 0.6
ROUGE_PRECISION_THRESHOLD = 0.3


# ==== Base class for all detectors ====
class BaseDetector(ABC):
    @abstractmethod
    def _predict(self, context: str, answer: str, output_format: str = "tokens") -> list:
        """Core prediction method to be implemented by subclasses."""
        pass


# ==== Transformer-based detector ====
class TransformerDetector(BaseDetector):
    def __init__(
        self,
        model_path: str,
        max_length: int = 4096,
        device=None,
        lang: Literal["en", "de", "fr", "es", "it", "pl"] = "en",
        **kwargs,
    ):
        """Initialize the TransformerDetector.

        :param model_path: The path to the model.
        :param max_length: The maximum length of the input sequence.
        :param device: The device to run the model on.
        :param lang: The language of the model.
        """
        if lang not in LANG_TO_PASSAGE:
            raise ValueError(f"Invalid language. Use one of: {', '.join(LANG_TO_PASSAGE.keys())}")

        self.lang = lang
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path, **kwargs)
        self.max_length = max_length
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        prompt_path = Path(__file__).parent.parent / "prompts" / f"qa_prompt_{lang.lower()}.txt"
        self.prompt_qa = Template(prompt_path.read_text(encoding="utf-8"))
        prompt_path = (
            Path(__file__).parent.parent / "prompts" / f"summary_prompt_{lang.lower()}.txt"
        )
        self.prompt_summary = Template(prompt_path.read_text(encoding="utf-8"))

    def _form_prompt(self, context: list[str], question: str | None) -> str:
        """Form a prompt from the provided context and question. We use different prompts for summary and QA tasks.

        :param context: A list of context strings.
        :param question: The question string.
        :return: The formatted prompt.
        """
        context_str = "\n".join(
            [
                f"{LANG_TO_PASSAGE[self.lang]} {i + 1}: {passage}"
                for i, passage in enumerate(context)
            ]
        )
        if question is None:
            return self.prompt_summary.substitute(text=context_str)
        else:
            return self.prompt_qa.substitute(
                question=question, num_passages=len(context), context=context_str
            )

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
        prompt = self._form_prompt(context, question)
        return self._predict(prompt, answer, output_format)


# ==== LLM-based detector ====
ANNOTATE_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "annotate",
            "description": "Return hallucinated substrings from the answer relative to the source.",
            "parameters": {
                "type": "object",
                "properties": {
                    "hallucination_list": {
                        "type": "array",
                        "items": {"type": "string"},
                    }
                },
                "required": ["hallucination_list"],
            },
        },
    }
]


class LLMDetector(BaseDetector):
    """LLM-powered hallucination detector using function calling and a prompt template."""

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: int = 0,
        lang: Literal["en", "de", "fr", "es", "it", "pl"] = "en",
        zero_shot: bool = False,
        fewshot_path: str | None = None,
        prompt_path: str | None = None,
        cache_file: str | None = None,
    ):
        """Initialize the LLMDetector.

        :param model: OpenAI model.
        :param temperature: model temperature.
        :param lang: language of the examples.
        :param zero_shot: whether to use zero-shot prompting.
        :param fewshot_path: path to the fewshot examples.
        :param prompt_path: path to the prompt template.
        :param cache_file: path to the cache file.
        """
        self.model = model
        self.temperature = temperature

        if lang not in LANG_TO_PASSAGE:
            raise ValueError(f"Invalid language. Use one of: {', '.join(LANG_TO_PASSAGE.keys())}")

        self.lang = lang
        self.zero_shot = zero_shot
        if fewshot_path is None:
            print(
                f"No fewshot path provided, using default path: {Path(__file__).parent.parent / 'prompts' / f'examples_{lang.lower()}.json'}"
            )
            fewshot_path = (
                Path(__file__).parent.parent / "prompts" / f"examples_{lang.lower()}.json"
            )

            if not fewshot_path.exists():
                raise FileNotFoundError(f"Fewshot file not found at {fewshot_path}")
        else:
            fewshot_path = Path(fewshot_path)

        if prompt_path is None:
            print(
                f"No prompt path provided, using default path: {Path(__file__).parent.parent / 'prompts' / 'hallucination_detection.txt'}"
            )
            template_path = Path(__file__).parent.parent / "prompts" / "hallucination_detection.txt"
        else:
            template_path = Path(prompt_path)

        prompt_qa_path = Path(__file__).parent.parent / "prompts" / f"qa_prompt_{lang.lower()}.txt"
        prompt_summary_path = (
            Path(__file__).parent.parent / "prompts" / f"summary_prompt_{lang.lower()}.txt"
        )

        self.template = Template(template_path.read_text(encoding="utf-8"))
        self.prompt_summary = Template(prompt_summary_path.read_text(encoding="utf-8"))

        self.fewshot = json.loads(fewshot_path.read_text(encoding="utf-8"))
        self.cache_path = cache_file

        if cache_file is None:
            self.cache_path = (
                Path(__file__).parent.parent / "cache" / f"cache_{self.model}_{self.lang}.json"
            )
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)

            # Read in cache
            if self.cache_path.exists():
                self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            else:
                self.cache = {}

            print(f"Cache file not provided, using default path: {self.cache_path}")
        else:
            self.cache_path = Path(cache_file)
            if not self.cache_path.exists():
                raise FileNotFoundError(f"Cache file not found at {self.cache_path}")
            self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))

    def _build_prompt(
        self,
        context: str,
        answer: str,
    ) -> str:
        """Fill the template with runtime values, inserting zero or many few‑shot examples.
        Uses `${placeholder}` tokens in the .txt file.
        """
        fewshot_block = ""
        if self.fewshot and not self.zero_shot:
            lines: list[str] = []
            for idx, ex in enumerate(self.fewshot, 1):
                lines.append(
                    f"""<example{idx}>
<source>{ex["source"]}</source>
<answer>{ex["answer"]}</answer>
<target>{{"hallucination_list": {json.dumps(ex["hallucination_list"], ensure_ascii=False)} }}</target>
</example{idx}>"""
                )
            fewshot_block = "\n".join(lines)

        filled = self.template.substitute(
            lang=self.lang,
            context=context,
            answer=answer,
            fewshot_block=fewshot_block,
        )
        return filled

    def _form_context(self, context: list[str], question: str | None) -> str:
        """Form a prompt from the provided context and question. We use different prompts for summary and QA tasks.
        :param context: A list of context strings.
        :param question: The question string.
        :return: The formatted prompt.
        """
        context_str = "\n".join(
            [f"passage {i + 1}: {passage}" for i, passage in enumerate(context)]
        )
        if question is None:
            return self.prompt_summary.substitute(text=context_str)
        else:
            return self.prompt_qa.substitute(
                question=question, num_passages=len(context), context=context_str
            )

    def _get_openai_client(self) -> OpenAI:
        """Get OpenAI client configured from environment variables.

        :return: Configured OpenAI client
        :raises ValueError: If API key is not set
        """
        api_key = os.getenv("OPENAI_API_KEY") or "EMPTY"
        api_base = os.getenv("OPENAI_API_BASE") or "https://api.openai.com/v1"

        return OpenAI(
            api_key=api_key,
            base_url=api_base,
        )

    def _hash(self, prompt: str) -> str:
        """Hash the prompt."""
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def _call_openai(self, prompt: str) -> str:
        """Call the OpenAI API.

        :param prompt: The prompt to call the OpenAI API with.
        :return: The response from the OpenAI API.
        """
        client = self._get_openai_client()
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in detecting hallucinations in LLM outputs.",
                },
                {"role": "user", "content": prompt},
            ],
            tools=ANNOTATE_SCHEMA,
            tool_choice={"type": "function", "function": {"name": "annotate"}},
            temperature=self.temperature,
        )

        return resp.choices[0].message.tool_calls[0].function.arguments

    def _save_cache(self):
        """Save the cache to the cache file."""
        self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False), encoding="utf-8")

    def _to_spans(self, subs: list[str], answer: str) -> list[dict]:
        """Convert a list of substrings to a list of spans.

        :param subs: A list of substrings.
        :param answer: The answer string.
        :return: A list of spans.
        """
        spans = []
        for s in subs:
            m = re.search(re.escape(s), answer)
            if m:
                spans.append({"start": m.start(), "end": m.end(), "text": s})
        return spans

    def _predict(self, context: str, answer: str, output_format: str = "spans") -> list:
        """Prompts the ChatGPT model to predict hallucination spans from the provided context and answer.

        :param context: The context string.
        :param answer: The answer string.
        :param output_format: works only for "spans" and returns grouped spans.
        """
        if output_format == "spans":
            llm_prompt = self._build_prompt(context, answer)

            key = self._hash("||".join([llm_prompt, self.model, str(self.temperature)]))

            # Check if the response is cached
            cached_response = self.cache.get(key)
            if cached_response is None:
                cached_response = self._call_openai(llm_prompt)
                self.cache[key] = cached_response
                self._save_cache()

            payload = json.loads(cached_response)
            return self._to_spans(payload["hallucination_list"], answer)
        else:
            raise ValueError(
                "Invalid output_format. This model can only predict hallucination spans. Use spans."
            )

    def predict_prompt(self, prompt: str, answer: str, output_format: str = "tokens") -> list:
        """Predict hallucination spans from the provided prompt and answer.

        :param prompt: The prompt string.
        :param answer: The answer string.
        :param output_format: "spans" to return grouped spans.
        """
        return self._predict(prompt, answer, output_format)

    def predict(
        self,
        context: list[str],
        answer: str,
        question: str | None = None,
        output_format: str = "spans",
    ) -> list:
        """Predict hallucination spans from the provided context, answer, and question.
        This is a useful interface when we don't want to predict a specific prompt, but rather we have a list of contexts, answers, and questions. Useful to interface with RAG systems.

        :param context: A list of context strings.
        :param answer: The answer string.
        :param question: The question string.
        :param output_format: "spans" to return grouped spans.
        """
        prompt = self._form_context(context, question)
        return self._predict(prompt, answer, output_format=output_format)


# ==== Rule-based detector ====
class RuleBasedDetector(BaseDetector):
    def __init__(
            self,
            lang: Literal["en", "de", "fr", "es", "it", "pl"] = "en",
                 ):
        """Initialize the RuleBasedDetector.

        :param lang: The language of the model.
        """
        if lang not in LANG_TO_PASSAGE:
            raise ValueError(f"Invalid language. Use one of: {', '.join(LANG_TO_PASSAGE.keys())}")

        self.lang = lang

        prompt_path = Path(__file__).parent.parent / "prompts" / f"qa_prompt_{lang.lower()}.txt"
        self.prompt_qa = Template(prompt_path.read_text(encoding="utf-8"))
        prompt_path = (
            Path(__file__).parent.parent / "prompts" / f"summary_prompt_{lang.lower()}.txt"
        )
        self.prompt_summary = Template(prompt_path.read_text(encoding="utf-8"))

    def _predict(self, context: list[str], answer: str, output_format: str = "spans") -> list:
        """
        Perform rule-based hallucination detection on the answer, comparing it to the provided context.

        :param context: A list of context strings to compare against.
        :param answer: The generated answer string to evaluate.
        :param output_format: Output format - either 'spans' (sentence-level) or 'tokens' (word-level).
        :return: A list of hallucination spans or tokens, each with confidence scores.
        """
        # Tokenize context for better comparison.
        context_str = " ".join(context).lower()
        number_hallucinations = self._detect_number_hallucinations(context_str, answer)
        context_sentences = [
            s.strip().lower()
            for ctx in context
            for s in re.split(r'(?<=[.?!])\s+', ctx)
            if s.strip()
        ]

        if output_format == "spans":
            spans = []
            # Iterate over all sentence-like segments in the answer.
            for match in re.finditer(r'[^.?!]+[.?!]', answer):
                sentence = match.group().strip()
                sentence_lower = sentence.lower()
                sentence_numbers = self._extract_numbers(sentence)

                fuzzy_match_score = self._fuzzy_sequence_matcher(sentence_lower, context_str)

                is_hallucinated = sentence_lower not in context_str and fuzzy_match_score < SEQUENCE_MATCH_THRESHOLD


                # Compare with each context sentence and store highest similarity
                # max_similarity = max(
                #     self._fuzzy_sequence_matcher(sentence_lower, ctx_sent)
                #     for ctx_sent in context_sentences
                # ) if context_sentences else 0.0

                has_number_hallucination = any(n in number_hallucinations for n in sentence_numbers)
                # is_hallucinated = max_similarity < SEQUENCE_MATCH_THRESHOLD


                if has_number_hallucination:
                    spans.append({
                        "text": sentence,
                        "start": match.start(),
                        "end": match.end(),
                        "confidence": 1
                    })
                elif is_hallucinated:
                    spans.append({
                        "text": sentence,
                        "start": match.start(),
                        "end": match.end(),
                        "confidence": 1 - fuzzy_match_score
                        # "confidence": 1 - max_similarity
                    })
            return spans

        elif output_format == "tokens":
            token_outputs = []
            # Check each word-level token in the answer.
            for match in re.finditer(r'\b\w+\b', answer):
                token = match.group()
                token_lower = token.lower()
                fuzzy_match_score = self._fuzzy_sequence_matcher(token_lower, context_str)

                is_hallucinated = token_lower not in context_str and fuzzy_match_score < SEQUENCE_MATCH_THRESHOLD 
                has_number_hallucination = token_lower in number_hallucinations

                # Check for number hallucination first, since variable is_hallucinated might be true as well in hallucinated number case
                if has_number_hallucination: 
                    token_outputs.append({
                        "token": token,
                        "pred": 1,
                        "prob": 1
                    })
                else:
                    token_outputs.append({
                        "token": token,
                        "pred": int(is_hallucinated),
                        "prob": 1 - fuzzy_match_score if is_hallucinated else 0.01
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
        return  Levenshtein.ratio(s1, s2)
    
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

        print("Entered predict method in Rule-based class")
        return self._predict(context, answer, output_format)
    
    def _extract_numbers(self, text: str) -> set:
        # This regex pattern checks first for numbers like 1,222.59; second for 0.12; third for 4,12; forth for common digits
        number_pattern = r'\d+\,\d+\.\d+|\d+\.\d+|\d+\,\d+|\d+'
        matches = re.findall(number_pattern, text)
        numbers = set()
        for m in matches:
            numbers.add(m)
        return numbers

    def _detect_number_hallucinations(self, context: str, answer: str) -> set:
        context_numbers = self._extract_numbers(context)
        answer_numbers = self._extract_numbers(answer)

        hallucinated = set()
        for num in answer_numbers:
           if not any((num == ctx_num) for ctx_num in context_numbers):
                hallucinated.add(num)

        return hallucinated


class RougeBasedDetector(BaseDetector):
    def __init__(
        self,
        lang: Literal["en", "de", "fr", "es", "it", "pl"] = "en",
    ):
        """
        Initialize the RougeBasedDetector.

        :param lang: The language of the model.
        """
        if lang != "en":
            raise NotImplementedError("Currently only English ('en') is supported for ROUGE-based detection.")
        
        nltk.download('punkt')
        nltk.download('stopwords')

        self.lang = lang
        self.stop_words = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()

    def _preprocess(self, text: str) -> list[str]:
        """Lowercase, tokenize, remove stopwords, and apply stemming."""
        tokens = word_tokenize(text.lower())

        return [
            self.stemmer.stem(token)
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




class HallucinationDetector:
    def __init__(self, method: str = "transformer", **kwargs):
        """Facade for the hallucination detector.

        :param method: "transformer" for the model-based approach, "rule" for the rule-based approach, "llm" for the LLM-based approach
        :param kwargs: Additional keyword arguments passed to the underlying detector.
        """
        if method == "transformer":
            self.detector = TransformerDetector(**kwargs)
        elif method == "rule":
            self.detector = RuleBasedDetector()
        elif method == "llm":
            self.detector = LLMDetector(**kwargs)
        elif method == "rouge":
            self.detector = RougeBasedDetector()
        else:
            raise ValueError("Unsupported method. Choose 'transformer', 'rule or 'llm'.")

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
        return self.detector.predict(context, answer, question, output_format)

    def predict_prompt(self, prompt: str, answer: str, output_format: str = "tokens") -> list:
        """Predict hallucination tokens or spans from the provided prompt and answer.

        :param prompt: The prompt string.
        :param answer: The answer string.
        """
        return self.detector.predict_prompt(prompt, answer, output_format)
# Models

## English Models

| Model | Base | Max Tokens | Example F1 | Span F1 |
|-------|------|-----------|-----------|---------|
| [lettucedetect-base-modernbert-en-v1](https://huggingface.co/KRLabsOrg/lettucedetect-base-modernbert-en-v1) | ModernBERT-base | 4K | 76.8% | SOTA |
| [lettucedetect-large-modernbert-en-v1](https://huggingface.co/KRLabsOrg/lettucedetect-large-modernbert-en-v1) | ModernBERT-large | 4K | 79.2% | SOTA |

## Multilingual Models

| Model | Base | Languages | Max Tokens |
|-------|------|-----------|-----------|
| [lettucedetect-base-eurobert-multilingual-v1](https://huggingface.co/KRLabsOrg/lettucedetect-base-eurobert-multilingual-v1) | EuroBERT-210M | en, de, fr, es, it, pl, cn | 8K |

## TinyLettuce (Distilled)

Smaller models for resource-constrained environments. See [TinyLettuce docs](../TINYLETTUCE.md).

## Using a Model

```python
from lettucedetect.models.inference import HallucinationDetector

detector = HallucinationDetector(
    method="transformer",
    model_path="KRLabsOrg/lettucedetect-large-modernbert-en-v1"
)
```

Models are downloaded automatically from HuggingFace Hub on first use.

# Multilingual Support

LettuceDetect supports hallucination detection in multiple languages via EuroBERT-based models.

## Supported Languages

| Language | Code | Model |
|----------|------|-------|
| English | en | ModernBERT + EuroBERT |
| German | de | EuroBERT |
| French | fr | EuroBERT |
| Spanish | es | EuroBERT |
| Italian | it | EuroBERT |
| Polish | pl | EuroBERT |
| Chinese | cn | EuroBERT |
| Hungarian | hu | EuroBERT |

## Usage

```python
from lettucedetect.models.inference import HallucinationDetector

detector = HallucinationDetector(
    method="transformer",
    model_path="KRLabsOrg/lettucedetect-base-eurobert-multilingual-v1"
)
```

The multilingual model handles all supported languages with a single checkpoint. No language detection or switching needed.

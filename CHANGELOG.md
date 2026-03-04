# Changelog

All notable changes to LettuceDetect are documented here.

## [Unreleased]

### Added
- Automatic context chunking for long inputs — when context exceeds `max_length`, passages are split into groups and scored independently with `max()` aggregation
- Hungarian language support (prompts and summary templates)
- GitHub Pages documentation site with KR Labs branding
- `CONTRIBUTING.md` for new contributors

### Fixed
- `answer_start_token` bug in `prepare_tokenized_input` — previously computed from context side, which gave wrong results when context was truncated; now computed from answer side
- `predict_prompt()` now warns when input exceeds `max_length` instead of silently truncating

### Changed
- CI lint job runs standalone (only needs ruff, no heavy deps)
- Modernized type hints across codebase (`list[str]` instead of `List[str]`)
- Replaced `print()` with `logging` in LLM detector
- Cleaned up duplicate test fixtures

## [0.1.8] - 2025-08-31

### Added
- RAGFactChecker integration for triplet-based hallucination detection
- Hallucination generation pipeline for synthetic training data
- LangChain integration (callbacks, chains, tools)
- Elysia framework integration
- TinyLettuce — smaller distilled model variants
- MkDocs documentation site
- Batch processing support for RAGFactChecker

## [0.1.7] - 2025-05-15

### Added
- Web API with FastAPI and async Python client (`lettucedetect_api/`)
- LLM-based hallucination detection using OpenAI API
- Multilingual support (German, French, Spanish, Italian, Polish, Chinese)
- RAGBench dataset preprocessing and training
- EuroBERT model support (8K context window)
- AUROC evaluation metric
- Caching for LLM API calls

### Changed
- Restructured detectors into factory pattern (`make_detector()`)
- Added seed to training for reproducibility

### Fixed
- Tensor copying bug in inference
- Encoder SEP token handling

## [0.1.6] - 2025-02-27

### Changed
- Migrated to `pyproject.toml` (removed setup.py)
- Added Ruff for linting and formatting
- Set up GitHub Actions CI/CD

## [0.1.5] - 2025-02-22

### Changed
- Improved token-level inference mapping

## [0.1.4] - 2025-02-12

### Changed
- README improvements

## [0.1.3] - 2025-02-12

### Added
- First interactive demo

## [0.1.2] - 2025-02-11

### Added
- `HallucinationDetector` interface — the main public API
- Character-level evaluation

## [0.1.1] - 2025-02-10

### Fixed
- Package distribution (included `preprocess` subpackage)

## [0.1.0] - 2025-02-09

### Added
- Initial release
- RAGTruth preprocessing pipeline
- First trained hallucination detection model (ModernBERT-base, 6 epochs)
- Token classification for hallucination span detection

[Unreleased]: https://github.com/KRLabsOrg/LettuceDetect/compare/0.1.8...HEAD
[0.1.8]: https://github.com/KRLabsOrg/LettuceDetect/compare/0.1.7...0.1.8
[0.1.7]: https://github.com/KRLabsOrg/LettuceDetect/compare/0.1.6...0.1.7
[0.1.6]: https://github.com/KRLabsOrg/LettuceDetect/compare/0.1.5...0.1.6
[0.1.5]: https://github.com/KRLabsOrg/LettuceDetect/compare/0.1.4...0.1.5
[0.1.4]: https://github.com/KRLabsOrg/LettuceDetect/compare/0.1.3...0.1.4
[0.1.3]: https://github.com/KRLabsOrg/LettuceDetect/compare/0.1.2...0.1.3
[0.1.2]: https://github.com/KRLabsOrg/LettuceDetect/compare/0.1.1...0.1.2
[0.1.1]: https://github.com/KRLabsOrg/LettuceDetect/compare/0.1.0...0.1.1
[0.1.0]: https://github.com/KRLabsOrg/LettuceDetect/releases/tag/0.1.0

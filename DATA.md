# Data Plan

The v1 data target is at least 500 MB of cleaned Algerian-centric text, ideally 1-2 GB. Roman-script/Arabizi text should stay at or above 25% of the final corpus.

## Manifest

The acquisition mission appends one JSON object to `data/manifest.jsonl` after every run. Each record includes:

- source name and input paths,
- output path and byte size,
- license or TOS status,
- allowed usage: `pretraining`, `evaluation`, `research-only`, or `restricted`,
- total, kept, filtered, and duplicate line counts,
- mean Arabic/Latin ratios and Arabizi token count.

Run the full Algerian Darija acquisition and cleaning mission:

```bash
python scripts/darija_data_mission.py
```

It writes raw source exports to `raw_data/`, cleaned JSONL/TXT to `cleaned_data/`, reports to `reports/`, and copies the final corpus to `FINAL_DARIJA_CORPUS/` only after the configured token target is reached. The report uses `cl100k_base` token counts and includes source breakdowns, script distribution, quality tiers, perplexity histogram, top words, and spot-check samples.

The source registry lives in `configs/darija_sources.json`. Active entries currently cover public/licensed Algerian Darija sources from Hugging Face and Zenodo, CAFE transcript extraction, and gated/manual entries for sources that require credentials or benchmark isolation.

Use:

```bash
python scripts/prepare_corpus.py \
  --input data/raw/youtube \
  --output corpus/darija_pretrain.txt \
  --source-name youtube_algerian_channels \
  --license-status "YouTube API, local research use" \
  --usage pretraining
```

## Source Priorities

1. DziriBERT comparison data: Twifil and NArabizi.
2. Public Algerian resources: DzNER, public Algerian sentiment datasets, NArabizi annotations.
3. Compliant local research scraping: YouTube API, tweet IDs/hydration, forums/news comments where terms allow.
4. Restricted resources such as ELNER-DZ stay marked `restricted` until access and usage rights are clear.

Do not mix evaluation test data into pretraining.

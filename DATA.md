# Data Plan

The v1 data target is at least 500 MB of cleaned Algerian-centric text, ideally 1-2 GB. Roman-script/Arabizi text should stay at or above 25% of the final corpus.

## Manifest

Every prepared source appends one JSON object to `data/manifest.jsonl` with:

- source name and input paths,
- output path and byte size,
- license or TOS status,
- allowed usage: `pretraining`, `evaluation`, `research-only`, or `restricted`,
- total, kept, filtered, and duplicate line counts,
- mean Arabic/Latin ratios and Arabizi token count.

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

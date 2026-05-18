# Benchmarks

The v1 gate is direct comparison with DziriBERT on Algerian tasks.

## Required Tasks

| Dataset | Task | Primary reason |
| --- | --- | --- |
| Twifil | sentiment | DziriBERT comparison in Arabic-script Algerian tweets |
| Twifil | emotion | DziriBERT comparison |
| NArabizi | sentiment | Roman-script/Arabizi strength test |
| NArabizi | topic | Roman-script/Arabizi strength test |

Report accuracy for DziriBERT comparability and macro-F1 for skew.

## Run

DziriBERT:

```bash
python scripts/evaluate_benchmarks.py \
  --task narabizi_sentiment \
  --model dziribert \
  --train data/narabizi/train.csv \
  --dev data/narabizi/dev.csv \
  --test data/narabizi/test.csv \
  --output runs/benchmarks/dziribert_narabizi_sentiment.json
```

SBERTa:

```bash
python scripts/evaluate_benchmarks.py \
  --task narabizi_sentiment \
  --model sberta \
  --sberta-checkpoint runs/darija-medium-v1/latest \
  --tokenizer-dir runs/tokenizer \
  --train data/narabizi/train.csv \
  --dev data/narabizi/dev.csv \
  --test data/narabizi/test.csv \
  --output runs/benchmarks/sberta_narabizi_sentiment.json
```

## Pass Rule

SBERTa passes v1 only if it beats DziriBERT on NArabizi sentiment/topic and is not worse on Twifil sentiment/emotion using the same splits and seeds.

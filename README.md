# EvidencePath-RAG

This repository contains an independent implementation of the paper
`EvidencePath-RAG: Edge-Based Relational Evidence Assembly for Context-Efficient Multi-Hop Question Answering`.

## Downloaded paper datasets

The raw development splits are stored locally under `data/raw/`:

| Benchmark | Paper split | Local file | Records |
| --- | --- | --- | ---: |
| HotpotQA | Distractor validation | `data/raw/hotpotqa/hotpot_dev_distractor_v1.json` | 7,405 |
| 2WikiMultiHopQA | Development | `data/raw/2wikimultihopqa/dev.json` | 12,576 |
| MuSiQue-Ans | Development | `data/raw/musique/musique_ans_v1.0_dev.jsonl` | 2,417 |

The HotpotQA file is byte-for-byte checksum-verified against the upstream file checksum. The downloaded files are excluded from Git because of their size. Dataset definitions and official download instructions are available from the [HotpotQA repository](https://github.com/hotpotqa/hotpot), [2WikiMultiHopQA repository](https://github.com/Alab-NII/2wikimultihop), and [MuSiQue repository](https://github.com/StonyBrookNLP/musique). The local snapshots came from [HotpotQA mirror](https://huggingface.co/datasets/RAGLAB/data/tree/83ec19575bcf252a53a82f68eda4a8921924ca45), [2Wiki development split](https://huggingface.co/datasets/voidful/2WikiMultihopQA/blob/16852fde9d85cba158cf7e6517e7a3f9415a28c0/dev.json), and [MuSiQue-Ans development split](https://huggingface.co/datasets/voidful/MuSiQue/blob/a7d9f9adf6191604fc67cde318ee1a86fcf7babc/musique_ans_v1.0_dev.jsonl).

The implementation includes:

- offline evidence-graph construction with sentence-aware chunks, entity-sharing edges, semantic bridge edges, and sentence-bearing edge embeddings;
- query-time routing with parallel-edge collapse, top-K seed retrieval, max activation propagation, adaptive hard-cap thresholds, candidate components, KMB Steiner approximation, and fallback cases;
- edge-only assembly with score-based greedy packing and a hard context-token cap;
- sentence retrieval, vanilla chunk retrieval, HippoRAG-style PPR, and self-contained graph/hierarchical baselines that use the same evaluation pipeline;
- HotpotQA/2WikiMultiHopQA/MuSiQue normalization, EM/F1, support recall, complete coverage, ablations, and paired bootstrap/Holm correction;
- scripts for data preparation, index construction, hard-cap/ablation/latency experiments, and result aggregation.

## Installation

```bash
pip install -e ".[all]"
```

`BAAI/bge-m3` and the answer generator are downloaded by Hugging Face when the scripts are run. `--encoder-backend hashing` is available for a small no-download smoke test; it is not intended to reproduce the paper's reported numbers.

## Reproduction workflow

The commands below are instructions only; this source tree does not execute them automatically.

### 1. Prepare evaluation questions and the corpus

```bash
python scripts/prepare_datasets.py \
  --dataset hotpotqa \
  --input data/raw/hotpotqa/hotpot_dev_distractor_v1.json \
  --output-dir artifacts/hotpotqa \
  --sample-size 5000 \
  --seed 42
```

The loader accepts local JSON/JSONL files or `--hf-dataset` when the `datasets` package is installed. The adapter preserves question IDs, answers, context documents, and support annotations. Use the same `questions.jsonl` for every budget and method.

For an open-domain corpus separate from question-local contexts, pass `--corpus-input path/to/corpus.jsonl`; otherwise the deduplicated question contexts are written as the corpus.

The other two paper splits are prepared with the same seed and the paper's sample sizes:

```bash
python scripts/prepare_datasets.py \
  --dataset 2wikimultihopqa \
  --input data/raw/2wikimultihopqa/dev.json \
  --output-dir artifacts/2wikimultihopqa \
  --sample-size 5000 \
  --seed 42

python scripts/prepare_datasets.py \
  --dataset musique-ans \
  --input data/raw/musique/musique_ans_v1.0_dev.jsonl \
  --output-dir artifacts/musique-ans \
  --sample-size 2417 \
  --seed 42
```

### 2. Build the reusable evidence index

```bash
python scripts/build_index.py \
  --corpus artifacts/hotpotqa/corpus.jsonl \
  --output artifacts/hotpotqa/index \
  --encoder-model BAAI/bge-m3 \
  --tokenizer-model BAAI/bge-m3 \
  --entity-backend spacy
```

When using `spacy`, install an appropriate NER model such as `en_core_web_sm`. Pass `--entity-backend regex` when no NER model is available.

### 3. Run the experiments

```bash
python scripts/run_experiments.py \
  --index artifacts/hotpotqa/index \
  --questions artifacts/hotpotqa/questions.jsonl \
  --output-dir artifacts/hotpotqa/results \
  --budgets 1536 2048 3072 \
  --methods evidencepath hipporag sentence linear raptor lightrag vanilla \
  --embedding-model BAAI/bge-m3 \
  --tokenizer-model Qwen/Qwen3.5-9B \
  --generator-model Qwen/Qwen3.5-9B \
  --quantize-8bit
```

Omit `--generator-model` to evaluate retrieval and support coverage only; predictions are then recorded as `Unknown`. `run_experiments.py` writes per-question JSONL files and summary JSON files without modifying the index.

Use `--routing-policy original --budgets 8192` for the paper's separate original-routing generator comparison. The hard-cap tables use the default `--routing-policy hardcap`.

Table 4 ablations:

```bash
python scripts/run_experiments.py \
  --index artifacts/hotpotqa/index \
  --questions artifacts/hotpotqa/questions.jsonl \
  --output-dir artifacts/hotpotqa/ablations \
  --budgets 2048 \
  --methods evidencepath \
  --ablations full no_propagation no_semantic_bridge no_steiner
```

Paired bootstrap and Holm correction:

```bash
python scripts/analyze_results.py \
  --results artifacts/hotpotqa/results \
  --output artifacts/hotpotqa/tables
```

# HeceTokenizer

A syllable-based tokenizer for Turkish NLP. HeceTokenizer exploits the deterministic six-pattern phonological structure of Turkish to construct a closed, OOV-free vocabulary of approximately 8,000 unique syllable types — no dictionary required.

## Key Results

HeceTokenizer achieves **50.3% Recall@5** on the TQuAD retrieval benchmark using a 1.5M parameter BERT-tiny model, surpassing a morphology-based baseline (TurkishTokenizer, Bayram et al. 2026) that uses a 300M parameter model.

| Method | Model Size | Recall@5 |
|--------|------------|----------|
| TurkishTokenizer (Bayram et al., 2026) | 300M | 46.92% |
| HeceTokenizer (chunk=8) | 1.5M | **50.3%** |

## How It Works

Turkish syllables follow exactly six phonological patterns:

```
V, CV, VC, CVC, VCC, CVCC
```

Any Turkish word can be unambiguously decomposed into syllables using these patterns via a simple right-to-left greedy algorithm. The resulting vocabulary is finite and closed — OOV tokens are theoretically impossible.

```
türkiye  →  tür + ki + ye
kardeş   →  kar + deş
trabzon  →  t + rab + zon
```

## Repository Contents

```
tok_hece.zip          — HuggingFace-compatible syllable tokenizer
model_hece_512_v1.pt  — BERT-tiny model trained on 26K Turkish Wikipedia articles
```

## Usage

```python
from transformers import PreTrainedTokenizerFast, BertConfig, BertForMaskedLM
import torch

# Load tokenizer
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="tok_hece/tokenizer.json",
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]"
)

# Load model
config = BertConfig(
    vocab_size=8000,
    hidden_size=128,
    num_hidden_layers=2,
    num_attention_heads=2,
    intermediate_size=512,
    max_position_embeddings=512,
    pad_token_id=0
)
model = BertForMaskedLM(config)
model.load_state_dict(torch.load("model_hece_512_v1.pt", map_location="cpu"))
model.eval()
```

## Training Details

| Parameter | Value |
|-----------|-------|
| Model | BERT-tiny (1.5M parameters) |
| Training data | 26,000 Turkish Wikipedia articles (~40% of Turkish Wikipedia) |
| Epochs | 10 |
| Hardware | NVIDIA T4 GPU |
| Max sequence length | 512 syllable tokens |
| Optimal chunk size | 8 syllable tokens (~2.6 words) |
| Mean pairwise cosine similarity | 0.257 |

## Citation

If you use this work, please cite:

```bibtex
@article{hecetokenizer2026,
  title   = {HeceTokenizer: A Syllable-Based Tokenization Approach for Turkish Retrieval},
  year    = {2026}
}
```

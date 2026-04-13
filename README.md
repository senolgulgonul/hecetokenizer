# HeceTokenizer

A syllable-based tokenizer for Turkish NLP. HeceTokenizer exploits the deterministic six-pattern phonological structure of Turkish to construct a closed, OOV-free vocabulary of approximately 8,000 unique syllable types — no dictionary, no training data required.

## Key Results

In a controlled comparison where architecture, training data, vocabulary size, and optimization procedure are held constant, HeceTokenizer achieves 65.9% Recall@5 on TQuAD, within 2.8 percentage points of BPE (68.7%), despite requiring no tokenizer training. Both methods substantially outperform larger morphology-based models.

| Method | Model Size | Recall@5 |
|--------|-----------|---------|
| TurkishTokenizer (Bayram et al., 2026) | 300M | 46.92% |
| Mursit (Bayram et al., 2026) | 300M | 35.43% |
| CosmosGPT2 (Bayram et al., 2026) | 300M | 33.81% |
| Tabi (Bayram et al., 2026) | 300M | 34.96% |
| BPE (chunk=3 words) | 1.5M | 68.7% |
| **HeceTokenizer (chunk=4 words)** | **1.5M** | **65.9%** |

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

Unlike BPE, HeceTokenizer preserves all Turkish-specific characters (ş, ğ, ü, ö, ç, ı), ensuring distinct words such as keş and kes receive different token sequences.

## Repository Contents

```
tok_hece.zip          — HuggingFace-compatible syllable tokenizer
model_hece_512_v1.pt  — BERT-tiny model trained on 26K Turkish Wikipedia articles
hecetokenizer.py      — Syllabification script (standalone, no dependencies)
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
ckpt = torch.load("model_hece_512_v1.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Syllabify and encode
from hecetokenizer import metni_hecele

text = "Türkiye büyük bir ülkedir"
syllabified = metni_hecele(text)  # "tür ki ye bü yük bir ül ke dir"
tokens = tokenizer(syllabified, return_tensors="pt")
```

## Retrieval Example

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from hecetokenizer import metni_hecele

def embed(texts, model, tokenizer, max_len=512):
    model.eval()
    vecs = []
    for text in texts:
        syl = metni_hecele(text)
        enc = tokenizer(syl, max_length=max_len, padding="max_length",
                        truncation=True, return_tensors="pt")
        with torch.no_grad():
            out = model.bert(**enc)
            vecs.append(out.last_hidden_state[:, 0, :].numpy())
    return np.vstack(vecs)

# Word-based sliding window chunking (optimal: 4 words)
def chunk_passage(passage, chunk_size=4):
    words = passage.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(len(words))]

passages = ["Türkiye Ankara'yı başkent olarak kabul eder.", "İstanbul Türkiye'nin en kalabalık şehridir."]
query = "Türkiye'nin başkenti neresidir?"

chunks = []
chunk_map = []
for i, p in enumerate(passages):
    for c in chunk_passage(p):
        chunks.append(c)
        chunk_map.append(i)

chunk_vecs = embed(chunks, model, tokenizer)
query_vec = embed([query], model, tokenizer)

scores = cosine_similarity(query_vec, chunk_vecs)[0]
top5 = np.argsort(scores)[::-1][:5]
retrieved_passages = set(chunk_map[i] for i in top5)
print([passages[i] for i in retrieved_passages])
```

## Training Details

| Parameter | Value |
|-----------|-------|
| Model | BERT-tiny (1.5M parameters) |
| Training data | 26,000 Turkish Wikipedia articles (~40% of Turkish Wikipedia) |
| Epochs | 10 |
| Hardware | NVIDIA T4 GPU |
| Max sequence length | 512 syllable tokens |
| Optimal chunk size | 4 words (~13 syllable tokens) |
| Mean pairwise cosine similarity | 0.257 |

## Citation

```bibtex
@article{hecetokenizer2026,
  title   = {HeceTokenizer: A Syllable-Based Tokenization Approach for Turkish Retrieval},
  author  = {Gulgonul, Senol},
  year    = {2026},
  url     = {https://arxiv.org/abs/...}
}
```

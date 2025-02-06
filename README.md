# **Simple GPT-2-like model**

This is a **simple GPT-2 style text generator** built with **PyTorch**. It reads a text file (`dataset.txt`), learns patterns in the text, and generates new text based on what it has learned.

---

## **Features**
- Uses **GPT-2 tokenization** (`tiktoken`).
- Implements **Transformer-based architecture** (self-attention, multi-head attention).
- Learns from **any text dataset** (`dataset.txt`).
- Supports **GPU acceleration** (if available).
- **Generates text** from a starting token.

---

## **Requirements**
Install dependencies using:
```bash
pip install torch numpy tqdm einops tiktoken

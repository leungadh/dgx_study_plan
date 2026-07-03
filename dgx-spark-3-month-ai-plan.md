# 3-Month AI / Deep Learning Project Plan for the NVIDIA DGX Spark

**Goal:** Go from "solid Transformer theory" to training, fine-tuning, quantizing, and serving LLMs on your own hardware — with a capstone that ties into your network-security expertise.

**Assumed starting point:** You've already studied self-attention, Q/K/V, causal masking, and worked through a minimal GPT (`gpt2_min.py`). This plan builds on that.

**Time budget:** ~8–12 hours/week. Each week has (a) study, (b) hands-on coding, (c) a deliverable.

---

## Know Your Hardware First (Weekend 0)

The DGX Spark is unusual — understand it before installing anything:

- **GB10 Grace Blackwell superchip**: 20-core ARM CPU + Blackwell GPU on one package
- **128 GB unified LPDDR5x memory** shared between CPU and GPU — no PCIe copies, no "CUDA out of memory" wall at 24GB like consumer cards
- **ARM64 (aarch64)** — many pip wheels don't exist; prefer NGC containers
- **DGX OS** (Ubuntu-based), CUDA 13, native FP4/NVFP4 support
- **~273 GB/s memory bandwidth** — modest for training throughput, but the huge capacity means you can *fit* 70B-class models for QLoRA fine-tuning and inference

**Setup checklist:**
1. Update DGX OS, verify `nvidia-smi` shows the GPU
2. Install Docker + NVIDIA Container Toolkit (usually pre-installed)
3. Pull the NGC PyTorch container: `docker pull nvcr.io/nvidia/pytorch:25.xx-py3` (use latest tag)
4. Bookmark **NVIDIA DGX Spark Playbooks**: https://build.nvidia.com/spark — official step-by-step guides for fine-tuning, vLLM, Ollama, ComfyUI etc. on this exact machine
5. Set up JupyterLab or VS Code Remote / Claude Code pointed at the Spark over SSH from your MacBook

---

## Month 1 — PyTorch Fluency + GPU Fundamentals

### Week 1: PyTorch tensors, autograd, and the training loop
- **Study:** PyTorch official "Learn the Basics" tutorial (https://pytorch.org/tutorials/beginner/basics/intro.html); Karpathy's *Neural Networks: Zero to Hero* videos 1–2 (micrograd, makemore) — https://karpathy.ai/zero-to-hero.html
- **Code:** Re-implement micrograd's backprop by hand; then rewrite it in PyTorch tensors on the GPU. Verify gradients match `torch.autograd`.
- **Deliverable:** A notebook: manual backprop vs autograd, running on CUDA, with timing comparison CPU vs GPU.

### Week 2: Training real models — MLPs and CNNs
- **Study:** PyTorch `nn.Module`, `DataLoader`, optimizers; Karpathy makemore parts 2–3 (MLP, BatchNorm)
- **Code:** Train an MLP language model on names data (makemore); train a small CNN on CIFAR-10. Practice: learning-rate finding, train/val curves, checkpointing.
- **Deliverable:** CIFAR-10 model >85% accuracy, with a clean reusable `train.py` you'll extend all quarter.

### Week 3: How the GPU actually works
This is where your systems background pays off — think of it like learning SRX SPU/NPU internals, but for CUDA.
- **Study:**
  - *Programming Massively Parallel Processors* (Hwu/Kirk/Hajj) chapters 1–5 — the canonical CUDA book
  - GPU MODE lecture series (formerly CUDA MODE): https://github.com/gpu-mode/lectures — lectures 1–3
  - Concepts: SMs, warps, threads/blocks/grids, global vs shared memory, memory coalescing, kernel launch
- **Code:** Write 2–3 tiny CUDA kernels (vector add, naive matmul, tiled matmul with shared memory). Compile with `nvcc` on the Spark. Then call a custom kernel from PyTorch via `torch.utils.cpp_extension` or write the same in **Triton** (https://triton-lang.org) — Triton is the friendlier path.
- **Deliverable:** Naive vs tiled matmul benchmark; short write-up of why tiling helps (memory hierarchy).

### Week 4: Profiling and performance
- **Study:** PyTorch Profiler tutorial; `nsys` (Nsight Systems) basics; mixed precision (`torch.autocast`, bf16); `torch.compile`
- **Code:** Profile your Week 2 CNN training. Find the bottleneck. Apply: bf16 autocast, larger batch (you have 128GB!), `torch.compile`, better DataLoader workers. Measure speedup at each step.
- **Deliverable:** A before/after profiling report — this skill transfers to everything later.

---

## Month 2 — Transformers From Scratch, Trained For Real

### Week 5: Rebuild GPT from scratch (no copying)
- **Study:** Re-watch Karpathy's *"Let's build GPT: from scratch"* (https://www.youtube.com/watch?v=kCc8FmEb1nY) but this time code along **from a blank file**; *The Illustrated Transformer* (https://jalammar.github.io/illustrated-transformer/) as reference; original paper *Attention Is All You Need*
- **Code:** Write your own `mingpt.py`: embedding, multi-head causal self-attention, MLP block, residuals, LayerNorm, weight tying. Train on tiny Shakespeare. Compare your implementation line-by-line against nanoGPT (https://github.com/karpathy/nanoGPT) only *after* it works.
- **Deliverable:** Your own GPT generating plausible Shakespeare, plus notes on every bug you hit.

### Week 6: Tokenizers + scaling up training
- **Study:** Karpathy's *"Let's build the GPT Tokenizer"* video; BPE algorithm; then nanoGPT's train loop internals (gradient accumulation, cosine LR schedule, weight decay, grad clipping)
- **Code:** Implement a minimal BPE tokenizer yourself; then train nanoGPT's GPT-2-small (124M) config on OpenWebText subset or FineWeb-Edu sample on the Spark. Use bf16 + `torch.compile`. It will be slower than an H100 — that's fine; run overnight and track loss curves with wandb (https://wandb.ai) or TensorBoard.
- **Deliverable:** A 124M model trained to a respectable validation loss, with logged curves.

### Week 7: Modern architecture upgrades
- **Study:** What changed since GPT-2: **RoPE** (rotary embeddings), **RMSNorm**, **SwiGLU**, **grouped-query attention (GQA)**, **KV cache** — read the Llama papers (Llama 1/2/3 arXiv) and https://github.com/meta-llama/llama for reference code; FlashAttention concept (Tri Dao's paper — you don't need to implement it, just understand IO-awareness)
- **Code:** Upgrade your Week 5 model: swap in RoPE, RMSNorm, SwiGLU. Implement a KV cache for generation and measure the speedup on long generations. Use `torch.nn.functional.scaled_dot_product_attention` (which dispatches to FlashAttention kernels) and benchmark vs your manual attention.
- **Deliverable:** "Llama-fied" mini model + KV-cache generation benchmark.

### Week 8: Consolidation + evaluation
- **Study:** How LLMs are evaluated: perplexity, HellaSwag, MMLU basics; Karpathy's *"Let's reproduce GPT-2"* video and https://github.com/karpathy/build-nanogpt for the full modern training recipe
- **Code:** Add a HellaSwag eval to your training run (build-nanogpt shows how). Do one clean end-to-end run: tokenize → train → eval → sample.
- **Deliverable:** A short "what I built" README for your repo. You now genuinely understand the full pretraining stack.

---

## Month 3 — Fine-Tuning, Quantization, Inference, and Capstone

This month leans into what the Spark is uniquely good at: big-memory fine-tuning and local serving.

### Week 9: Fine-tuning with LoRA/QLoRA
- **Study:** LoRA paper (arXiv:2106.09685) — the math is simple and elegant; QLoRA paper (arXiv:2305.14314); Hugging Face PEFT docs (https://huggingface.co/docs/peft) and the HF LLM Course (https://huggingface.co/learn/llm-course)
- **Code:** First implement LoRA *by hand* on your own mini-GPT (it's ~30 lines — replace a Linear with W + BA). Then use PEFT + `transformers` (or **Unsloth**, https://github.com/unslothai/unsloth, which has DGX Spark support) to QLoRA fine-tune **Llama 3.1 8B** on an instruction dataset. With 128GB you can also attempt a 70B QLoRA — an experiment almost no consumer hardware can do.
- **Deliverable:** A fine-tuned 8B model that follows your custom instruction format.

### Week 10: Quantization + local inference engines
- **Study:** Quantization concepts: int8, GPTQ/AWQ, GGUF k-quants, and Blackwell's native **NVFP4**; how inference servers work: continuous batching, paged attention (vLLM paper)
- **Code:** Run the same model through three stacks and benchmark tokens/sec + quality:
  1. **llama.cpp / Ollama** (GGUF) — https://github.com/ggml-org/llama.cpp
  2. **vLLM** — https://github.com/vllm-project/vllm (use the DGX Spark playbook)
  3. **TensorRT-LLM** with NVFP4 if you're feeling ambitious
- **Deliverable:** A comparison table: engine × quantization × tok/s × quality notes. Stand up one of them as a persistent OpenAI-compatible API on your home network.

### Weeks 11–12: Capstone — a Network-Security Domain Assistant
This is where your day job and your SRX/5G study notes become an asset:
- **Build a RAG + fine-tuned assistant for network security.** Corpus: your Juniper SRX notes, 5G core security material (N1–N13, GTP, PFCP), Junos docs, MITRE ATT&CK mappings.
- Components:
  1. **RAG pipeline**: chunk documents, embed with a local embedding model (e.g. `bge-m3` or `nomic-embed`), store in a vector DB (Chroma or LanceDB), retrieve + answer with your locally served model
  2. **Optional fine-tune**: QLoRA the model on Q&A pairs generated from your notes so it "speaks Junos" natively
  3. **Eval set**: 30–50 questions you know the answers to (e.g. "which SRX screen option mitigates X", "what does N4/PFCP expose") — score RAG vs fine-tune vs both
- **Deliverable:** A working local assistant you can query from your Mac, plus a write-up. This is genuinely portfolio-worthy and useful at work.

---

## Reference Stack (bookmark these)

| Category | Resource |
|---|---|
| Spark-specific | NVIDIA DGX Spark Playbooks — build.nvidia.com/spark |
| PyTorch | pytorch.org/tutorials, *Deep Learning with PyTorch* (Stevens et al.) |
| From-scratch LLMs | Karpathy Zero-to-Hero, nanoGPT, build-nanogpt; Sebastian Raschka's *Build a Large Language Model (From Scratch)* + github.com/rasbt/LLMs-from-scratch |
| GPU/CUDA | *Programming Massively Parallel Processors*; GPU MODE lectures; Triton docs |
| Fine-tuning | HF PEFT + LLM course; Unsloth; LoRA/QLoRA papers |
| Inference | llama.cpp, Ollama, vLLM docs, TensorRT-LLM |
| Papers | Attention Is All You Need; GPT-2/3; Llama 1–3; LoRA; QLoRA; FlashAttention; vLLM (PagedAttention) |

## Habits That Make It Stick
- Keep **one git repo** for the whole quarter; commit every session
- Keep a **lab journal** (markdown) — bugs hit, fixes, benchmark numbers
- **Type code yourself** — never copy-paste tutorial code; the bugs are the learning
- End each week by explaining one concept out loud (or to Claude) with no notes — Feynman technique

## Stretch Ideas (if you finish early or want a different angle)
- **ML for network security**: train an anomaly-detection model on NetFlow/packet-capture data — directly relevant to your SRX/5G work
- **Speculative decoding**: implement draft-model speculative decoding on your local stack
- **Whisper + video**: use the Spark to transcribe/caption your England tour footage locally as an ffmpeg pipeline side-quest
- **Diffusion detour**: run ComfyUI / Flux on the Spark (there's an official playbook) to see the other half of generative AI

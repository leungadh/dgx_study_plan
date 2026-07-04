# DGX Spark 3-Month AI Study Plan

A self-directed, 12-week curriculum for going from solid Transformer theory to **training, fine-tuning, quantizing, and serving LLMs** on an NVIDIA DGX Spark — plus a web-based tracker to keep the quarter on rails.

The capstone ties it back to my day job: a locally served **RAG + fine-tuned network-security assistant** built over Juniper SRX and 5G core security notes, red-teamed against the OWASP LLM Top 10.

## What's in the repo

| File | Purpose |
|---|---|
| [`dgx-spark-3-month-ai-plan.md`](dgx-spark-3-month-ai-plan.md) | The full week-by-week plan: study material, hands-on exercises, and a deliverable per week |
| [`overview.md`](overview.md) | A short summary of the plan's three arcs |
| [`index.html`](index.html) | The progress tracker — a single self-contained page, no build step, no dependencies |
| [`render.yaml`](render.yaml) | Render.com Blueprint that deploys the repo as a free static site |

## The plan at a glance

- **Weekend 0 — Know your hardware.** The DGX Spark is unusual: GB10 Grace Blackwell superchip, 128 GB unified memory, ARM64 (prefer NGC containers over pip), native NVFP4.
- **Month 1 — PyTorch fluency + GPU fundamentals.** Tensors and autograd through CUDA/Triton kernels, then profiling and performance work (bf16, `torch.compile`, activation checkpointing).
- **Month 2 — Transformers from scratch, trained for real.** Rebuild GPT from a blank file, write a BPE tokenizer, pretrain a 124M model on the Spark, then modernize it: RoPE, RMSNorm, SwiGLU, KV cache, sampling strategies — with MoE/MLA study for what came after dense Llama 3.
- **Month 3 — Fine-tuning, quantization, inference, capstone.** LoRA by hand, QLoRA on 8B (and maybe 70B — the 128 GB flex), DPO, then benchmark llama.cpp vs vLLM vs TensorRT-LLM and stand up a local OpenAI-compatible API.
- **Weeks 11–12 — Capstone.** The network-security assistant: hybrid retrieval (dense + BM25 + reranker), optional domain QLoRA, a 30–50-question eval set, and a prompt-injection red-team of the finished system.

## The tracker

A zero-dependency progress dashboard for the plan: every task as a checkbox with Study / Code / **Milestone** badges, per-week lab-journal notes, stat tiles, and progress meters. Light and dark mode.

- **Run locally:** just open `index.html` in a browser.
- **Persistence:** progress and notes live in the browser's `localStorage`; use **Export / Import (JSON)** to back up or move between devices.
- **Editing the plan:** tasks are defined in the `PLAN` array at the top of the inline script. Give new tasks fresh IDs — saved progress is keyed by task ID.

## Deploying the tracker (free)

**Render.com:** *New → Blueprint*, point it at this repo — `render.yaml` configures a static site publishing the repo root. Every push to `main` redeploys. (Or *New → Static Site* with an empty build command and `.` as the publish directory.)

**GitHub Pages works too:** *Settings → Pages → Deploy from branch* (`main`, root).

## Habits that make it stick

One git repo, commit every session · keep a lab journal · type code yourself, never paste · end each week explaining one concept aloud, Feynman-style.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A 3-month AI/deep-learning study plan for the NVIDIA DGX Spark, plus a static web tracker for it. Pushed to https://github.com/leungadh/dgx_study_plan.

- `dgx-spark-3-month-ai-plan.md` — the full week-by-week plan (Month 1: PyTorch + CUDA/Triton fundamentals; Month 2: GPT pretraining from scratch, then modernizing with RoPE/RMSNorm/SwiGLU/KV cache; Month 3: LoRA/QLoRA fine-tuning, quantization, inference-engine benchmarking, and a capstone RAG + fine-tuned network-security assistant)
- `overview.md` — a short summary of the plan's arcs and practical tips
- `index.html` — the progress tracker: a single self-contained file (vanilla JS, no build step, no dependencies). The plan's tasks live in the `PLAN` array in its inline script; progress/notes persist to localStorage under key `dgx-tracker-v1` with JSON export/import. If the plan documents change, update `PLAN` to match. Test by opening the file in a browser — check both light and dark (`prefers-color-scheme`) modes.
- `render.yaml` — Render.com Blueprint deploying the repo root as a free static site

## Context that matters when helping here

- The user's background is network security (Juniper SRX, 5G core security — N1–N13, GTP, PFCP) with solid systems knowledge; they already understand Transformer theory (self-attention, Q/K/V, causal masking) and have worked through a minimal GPT. Pitch explanations accordingly — systems analogies land well, ML basics don't need re-teaching.
- Target hardware is the DGX Spark: GB10 Grace Blackwell, 20-core ARM CPU, 128 GB unified memory, ~273 GB/s bandwidth, DGX OS, CUDA 13, native NVFP4. Key implications: ARM64 means many pip wheels don't exist — prefer NGC containers (`nvcr.io/nvidia/pytorch:25.xx-py3`); large memory capacity favors big-model QLoRA/inference over raw training throughput.
- NVIDIA publishes DGX Spark playbooks (build.nvidia.com/spark) with official recipes for fine-tuning, vLLM, Ollama, etc. — check there before improvising setup steps.
- The plan emphasizes typing code from scratch rather than copy-pasting tutorial code; when helping with plan exercises, favor guidance and debugging help over writing complete solutions.
- Actual project code from the plan (training scripts, kernels, the capstone) is intended to live in a separate quarter-long git repo, not here. Edits here are edits to the plan documents themselves.

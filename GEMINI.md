# TurboQuant Mac — Project Index

## ⚙️ Локальный контекст (Правила и Стек)
Вся инфраструктура и жесткие архитектурные правила вынесены в локальную память:
- [stack.md](./.gemini/rules/stack.md) — Стек проекта.
- [constraints.md](./.gemini/rules/constraints.md) — Критические ограничения.



## 📌 Context
- **What it is:** Pure PolarQuant algorithm for extreme LLM KV Cache compression on Apple Silicon, with Asymmetric K/V scaling & Boundary V isolation. Note: QJL was dropped as it increased variance.
- **Stack:** Python 3, Apple MLX (`mlx.core`, Metal GPU), `mlx_lm`.

## 🛑 Critical Constraints / Red Flags
- Custom `.metal` shaders are currently forbidden (lazy graph compilation is sufficient, stay in Python).
- OOM Risk: Mac OOM kills 70B layer-by-layer streaming on 16GB. Do not brute force it without proper disk mmap chunking.

## 🚀 Environment (Quick Start)
- **Install:** `pip install -e .`
- **Prod Server:** `python3 scripts/run_server.py --model mlx-community/Meta-Llama-3-8B-Instruct-4bit`

## 📚 Documentation Navigation
- 🏗 **Architecture & Monkey-patching Details:** `docs/architecture.md`
- 📦 **Scripts & Tests Setup:** `docs/deploy.md`
- 📅 **Active Sprint & Next Steps:** `docs/sprints/week-1.md`

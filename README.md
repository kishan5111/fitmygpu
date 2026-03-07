# FitMyGPU

FitMyGPU helps you estimate whether a model will fit on a specific GPU before you download or run it.
It focuses on clear VRAM estimates, readable memory breakdowns, and model-aware notes that explain why a checkpoint behaves the way it does.
The goal is a calm technical tool that answers the fit question quickly without turning into a dashboard.

## Preview

![FitMyGPU UI](./docs/ui-page-home.png)

## About The Project

FitMyGPU is a focused web tool for one question: will a selected model fit on a selected GPU?

Instead of acting like a dashboard, it stays narrow and useful. You pick a model, dtype, and GPU, and the app estimates the memory footprint for inference or training, shows where the memory goes, and gives a fast fit verdict with practical suggestions.

The project is intentionally model-aware. Each supported model can carry architecture details, a research highlight, and a memory note so the UI teaches something real about why that checkpoint behaves the way it does in memory.

## Features

- Clean single-page VRAM calculator with a calm, high-trust UI
- Deterministic memory estimates for inference and training workflows
- Memory breakdown across weights, KV cache, activations, gradients, optimizer state, and overhead
- Compact formula view that explains how the estimate is calculated
- Model-specific notes via `researchHighlight` and `memoryNote`
- Shareable URL state for calculator inputs and results
- Curated local model registry instead of a noisy unfiltered model list

## Model Coverage

The registry is designed to keep growing. As new model families and checkpoints are added, FitMyGPU will also update the model-specific notes, architecture highlights, and memory guidance for them.

This is especially important for newer releases where architecture choices directly affect VRAM behavior, such as MoE routing, grouped KV heads, FP8 checkpoints, or compressed weight formats.

## Support

If this project is useful to you, support it here:

[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20me%20a%20coffee-support-FFDD00?style=for-the-badge&logo=buymeacoffee&logoColor=000000)](https://buymeacoffee.com/kishanvavdara)
# fitmygpu

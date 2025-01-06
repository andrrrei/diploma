# Methods for fine-tuning LLMs on instructions to improve performance in Russian

This repository contains the report for my pre-graduation practical research on fine-tuning LLMs to improve their performance on Russian language tasks. The research focuses on aligning models with user preferences using SFT and explores other advanced optimization techniques.

## Overview

### Objective
The goal of this project is to study and apply modern methods for fine-tuning large language models to enhance their quality for tasks involving the Russian language. Key methods include:
- Supervised Fine-Tuning (SFT)
- Reinforcement Learning from Human Feedback (RLHF)
- Direct Preference Optimization (DPO)
- Simple Preference Optimization (SimPO)
- Kahneman-Tversky Optimization (KTO)

### Highlights
- **Model Used**: Qwen2.5-3B, adapted for the Russian language.
- **Datasets**: 
  - **GrandMaster-PRO-MAX**: 156,000 instruction-response pairs, synthesized using GPT-4-Turbo.
  - **Saiga Scored**: 36,000 pairs with quality scores.
- **Evaluation Metrics**: LLM-as-a-Judge and manual review.
- **Technologies**: Hugging Face Transformers, LoRA (Low-Rank Adaptation), and parameter-efficient tuning methods.

## Experiments
Several experiments were conducted to optimize model alignment and evaluate performance:
1. Fine-tuning on subsets of GrandMaster-PRO-MAX to analyze data volume effects.
2. Tuning on Saiga Scored to understand the impact of high-quality but shorter responses.
3. Combining both datasets for balanced fine-tuning.
4. Exploring token length filtering to control response quality and length.
5. Comparing datasets' influence on response quality and alignment.

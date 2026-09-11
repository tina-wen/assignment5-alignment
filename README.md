# CS336 Spring 2026 Assignment 5: Alignment

## 概述

- 针对作业材料的学习笔记请参见 [`notes.pdf`](https://github.com/tina-wen/assignment5-alignment/blob/main/notes.pdf)。
- 作业 5 的问题分为以下三类：
  1. **组件实现**：实现 RL 算法推理生成之后的各个组件，包括 response 评分、计算优势、计算损失和梯度累积、模型参数更新循环等。此类问题已有明确的测试用例，使用 `uv run pytest` 即可快速检验正确性。
  2. **完整脚本与实验分析**：实现完整的 RL 算法脚本，调整超参数观察算法效果，并给出评估结论。
  3. **数理推导**：RL 算法相关的数学推导。
- 原作业仓库提供了：
  - [GSM8K 数据集](https://github.com/tina-wen/assignment5-alignment/tree/main/data/gsm8k)；
  - [三种 prompt 模板](https://github.com/tina-wen/assignment5-alignment/tree/main/cs336_alignment/prompts)；
  - [两种推理评测函数](https://github.com/tina-wen/assignment5-alignment/blob/main/cs336_alignment/drgrpo_grader.py)。

## 实现与交付

- 针对第 1 类问题，全部函数实现在 [`function.py`](https://github.com/tina-wen/assignment5-alignment/blob/main/function.py) 中，并通过 [`adapters.py`](https://github.com/tina-wen/assignment5-alignment/blob/main/tests/adapters.py) 的测试接口进行转发。
- 针对第 2、3 类问题，主要实现参考 `grpo_experiments*.py`，评估结果请参见 `writeup.pdf`(to be updated)。

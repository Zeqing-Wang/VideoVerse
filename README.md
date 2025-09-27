# <img src="assets/logo.png" width="60px" align="center"> VideoVerse: How Far is Your T2V Generator from a World Model?

Official repository for the paper ["VideoVerse: How Far is Your T2V Generator from a World Model?"](https://www.arxiv.org/abs/2506.02161).

[🌐 Webpage](https://www.naptmn.cn/Homepage_of_VideoVerse/) [📖 Paper](https://www.arxiv.org/abs/2506.02161) [🤗 Huggingface Dataset](https://huggingface.co/datasets/NNaptmn/VideoVerse) [🏆 Leaderboard](https://www.naptmn.cn/Homepage_of_VideoVerse/#leaderboard)

## 🔥 News
- **[2025.09.29]** 🔥 TBD

## Introduction


## 🔧 How to Start


VideoVerse is organized for easy benchmarking of T2V models:

### 📑 Prompt Files


### Generate Images & Directory Organization

### 🧪 Evaluation with VLM

1. **Set variables:**
   
2. **Run evaluation:**
   
3. **Summarize results:**
   

### 📊 Evaluation Results


1. **Start service:**
   ```bash
   pip install vllm
   vllm serve --model checkpoints/Qwen2.5-VL-72B-Instruct --port 8000 --host 0.0.0.0 --dtype bfloat16
   ```
2. **Update your evaluation command to use the Qwen2.5-VL endpoint.**

### 📜 Evaluation of Text Rendering

1. **Preparation:**
   
2. **Run evaluation:**
   

## 📣 Citation

```

## 🙋‍♂️ Questions?

Open an [issue](https://github.com/Zeqing-Wang/VideoVerse/issues) or start a [discussion](https://github.com/Zeqing-Wang/VideoVerse/discussions).

**Enjoy using VideoVerse!** 🚀🖼️🤖

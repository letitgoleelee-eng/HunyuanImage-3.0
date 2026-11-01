
## 📖 Introduction

**HunyuanImage-3.0** is a groundbreaking native multimodal model that unifies multimodal understanding and generation within an autoregressive framework. Our text-to-image module achieves performance **comparable to or surpassing** leading closed-source models.


<div align="center">
  <img src="./assets/framework.png" alt="HunyuanImage-3.0 Framework" width="90%">
</div>

## ✨ Key Features

* 🧠 **Unified Multimodal Architecture:** Moving beyond the prevalent DiT-based architectures, HunyuanImage-3.0 employs a unified autoregressive framework. This design enables a more direct and integrated modeling of text and image modalities, leading to surprisingly effective and contextually rich image generation.

* 🏆 **The Largest Image Generation MoE Model:** This is the largest open-source image generation Mixture of Experts (MoE) model to date. It features 64 experts and a total of 80 billion parameters, with 13 billion activated per token, significantly enhancing its capacity and performance.

* 🎨 **Superior Image Generation Performance:** Through rigorous dataset curation and advanced reinforcement learning post-training, we've achieved an optimal balance between semantic accuracy and visual excellence. The model demonstrates exceptional prompt adherence while delivering photorealistic imagery with stunning aesthetic quality and fine-grained details.

* 💭 **Intelligent World-Knowledge Reasoning:** The unified multimodal architecture endows HunyuanImage-3.0 with powerful reasoning capabilities. It leverages its extensive world knowledge to intelligently interpret user intent, automatically elaborating on sparse prompts with contextually appropriate details to produce superior, more complete visual outputs.


## 🛠️ Dependencies and Installation

### 💻 System Requirements

* 🖥️ **Operating System:** Linux
* 🎮 **GPU:** NVIDIA GPU with CUDA support
* 💾 **Disk Space:** 170GB for model weights
* 🧠 **GPU Memory:** ≥3×80GB (4×80GB recommended for better performance)

### 📦 Environment Setup

* 🐍 **Python:** 3.12+ (recommended and tested)
* 🔥 **PyTorch:** 2.7.1
* ⚡ **CUDA:** 12.8

### 📥 Install Dependencies

```bash
# 1. First install PyTorch (CUDA 12.8 Version)
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# 2. Then install tencentcloud-sdk
pip install -i https://mirrors.tencent.com/pypi/simple/ --upgrade tencentcloud-sdk-python

# 3. Then install other dependencies
pip install -r requirements.txt
```

#### Performance Optimizations

For **up to 3x faster inference**, install these optimizations:

```bash
# FlashAttention for faster attention computation
pip install flash-attn==2.8.3 --no-build-isolation

# FlashInfer for optimized moe inference. v0.3.1 is tested.
pip install flashinfer-python
```
> 💡**Installation Tips:** It is critical that the CUDA version used by PyTorch matches the system's CUDA version. 
> FlashInfer relies on this compatibility when compiling kernels at runtime. Pytorch 2.7.1+cu128 is tested.
> GCC version >=9 is recommended for compiling FlashAttention and FlashInfer.

> ⚡ **Performance Tips:** These optimizations can significantly speed up your inference!

> 💡**Notation:** When FlashInfer is enabled, the first inference may be slower (about 10 minutes) due to kernel compilation. Subsequent inferences on the same machine will be much faster.

## 🚀 Usage

### 🔥 Quick Start with Transformers

#### 1️⃣ Download model weights

```bash
# Download from HuggingFace and rename the directory.
# Notice that the directory name should not contain dots, which may cause issues when loading using Transformers.
hf download tencent/HunyuanImage-3.0 --local-dir ./HunyuanImage-3
```

#### 2️⃣ Run with Transformers

```python
from transformers import AutoModelForCausalLM

# Load the model
model_id = "./HunyuanImage-3"
# Currently we can not load the model using HF model_id `tencent/HunyuanImage-3.0` directly 
# due to the dot in the name.

kwargs = dict(
    attn_implementation="sdpa",     # Use "flash_attention_2" if FlashAttention is installed
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto",
    moe_impl="eager",   # Use "flashinfer" if FlashInfer is installed
)

model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
model.load_tokenizer(model_id)

# generate the image
prompt = "A brown and white dog is running on the grass"
image = model.generate_image(prompt=prompt, stream=True)
image.save("image.png")
```

### 🏠 Local Installation & Usage

#### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Tencent-Hunyuan/HunyuanImage-3.0.git
cd HunyuanImage-3.0/
```

#### 2️⃣ Download Model Weights

```bash
# Download from HuggingFace
hf download tencent/HunyuanImage-3.0 --local-dir ./HunyuanImage-3
```

#### 3️⃣ Run the Demo
The Pretrain Checkpoint does not automatically rewrite or enhance input prompts, for optimal results currently, we recommend community partners to use deepseek to rewrite the prompts. You can go to [Tencent Cloud](https://cloud.tencent.com/document/product/1772/115963#.E5.BF.AB.E9.80.9F.E6.8E.A5.E5.85.A5) to apply for an API Key.

```bash
# set env
export DEEPSEEK_KEY_ID="your_deepseek_key_id"
export DEEPSEEK_KEY_SECRET="your_deepseek_key_secret"

python3 run_image_gen.py --model-id ./HunyuanImage-3 --verbose 1 --sys-deepseek-prompt "universal" --prompt "A brown and white dog is running on the grass"
```

#### 4️⃣ Command Line Arguments

| Arguments               | Description                                                  | Default     |
| ----------------------- | ------------------------------------------------------------ | ----------- |
| `--prompt`              | Input prompt                                                 | (Required)  |
| `--model-id`            | Model path                                                   | (Required)  |
| `--attn-impl`           | Attention implementation. Either `sdpa` or `flash_attention_2`. | `sdpa`      |
| `--moe-impl`            | MoE implementation. Either `eager` or `flashinfer`           | `eager`     |
| `--seed`                | Random seed for image generation                             | `None`      |
| `--diff-infer-steps`    | Diffusion infer steps                                        | `50`        |
| `--image-size`          | Image resolution. Can be `auto`, like `1280x768` or `16:9`   | `auto`      |
| `--save`                | Image save path.                                             | `image.png` |
| `--verbose`             | Verbose level. 0: No log; 1: log inference information.      | `0`         |
| `--rewrite`             | Whether to enable rewriting                                  | `1`         |
| `--sys-deepseek-prompt` | Select sys-prompt from `universal` or `text_rendering`       | `universal` |

### 🎨 Interactive Gradio Demo

Launch an interactive web interface for easy text-to-image generation.

#### 1️⃣ Install Gradio

```bash
pip install gradio>=4.21.0
```

#### 2️⃣ Configure Environment

```bash
# Set your model path
export MODEL_ID="path/to/your/model"

# Optional: Configure GPU usage (default: 0,1,2,3)
export GPUS="0,1,2,3"

# Optional: Configure host and port (default: 0.0.0.0:443)
export HOST="0.0.0.0"
export PORT="443"
```

#### 3️⃣ Launch the Web Interface

**Basic Launch:**
```bash
sh run_app.sh
```

**With Performance Optimizations:**
```bash
# Use both optimizations for maximum performance
sh run_app.sh --moe-impl flashinfer --attn-impl flash_attention_2
```

#### 4️⃣ Access the Interface

> 🌐 **Web Interface:** Open your browser and navigate to `http://localhost:443` (or your configured port)


## 🧱 Models Cards

| Model                     | Params | Download | Recommended VRAM | Supported |
|---------------------------| --- | --- | --- | --- |
| HunyuanImage-3.0          | 80B total (13B active) | [HuggingFace](https://huggingface.co/tencent/HunyuanImage-3.0) | ≥ 3 × 80 GB | ✅ Text-to-Image
| HunyuanImage-3.0-Instruct | 80B total (13B active) | [HuggingFace](https://huggingface.co/tencent/HunyuanImage-3.0-Instruct) | ≥ 3 × 80 GB | ✅ Text-to-Image<br>✅ Prompt Self-Rewrite <br>✅ CoT Think



Notes:
- Install performance extras (FlashAttention, FlashInfer) for faster inference.
- Multi‑GPU inference is recommended for the Base model.




## 📊 Evaluation

* 🤖 **SSAE (Machine Evaluation)**   
SSAE (Structured Semantic Alignment Evaluation) is an intelligent evaluation metric for image-text alignment based on advanced multimodal large language models (MLLMs). We extracted 3500 key points across 12 categories, then used multimodal large language models to automatically evaluate and score by comparing the generated images with these key points based on the visual content of the images. Mean Image Accuracy represents the image-wise average score across all key points, while Global Accuracy directly calculates the average score across all key points.

<p align="center">
  <img src="./assets/ssae_side_by_side_comparison.png" width=98% alt="Human Evaluation with Other Models">
</p>

<p align="center">
  <img src="./assets/ssae_side_by_side_heatmap.png" width=98% alt="Human Evaluation with Other Models">
</p>


* 👥 **GSB (Human Evaluation)** 

We adopted the GSB (Good/Same/Bad) evaluation method commonly used to assess the relative performance between two models from an overall image perception perspective. In total, we utilized 1,000 text prompts, generating an equal number of image samples for all compared models in a single run. For a fair comparison, we conducted inference only once for each prompt, avoiding any cherry-picking of results. When comparing with the baseline methods, we maintained the default settings for all selected models. The evaluation was performed by more than 100 professional evaluators. 

<p align="center">
  <img src="./assets/gsb.png" width=98% alt="Human Evaluation with Other Models">
</p>


## 📚 Citation

If you find HunyuanImage-3.0 useful in your research, please cite our work:

```bibtex
@misc{HunyuanImage-3.0,
  title={HunyuanImage 3.0: Technical Report},
  author={Tencent Hunyuan Team},
  year={2025},
  howpublished={\url{https://github.com/Tencent-Hunyuan/HunyuanImage-3.0}},
}
```



<div align="center">

  <h1>
    <img src="assets/Stand-In.png" width="85" alt="Logo" valign="middle">
    Stand-In
  </h1>

  <h3>A Lightweight and Plug-and-Play Identity Control for Video Generation</h3>



[![arXiv](https://img.shields.io/badge/arXiv-2508.07901-b31b1b)](https://arxiv.org/abs/2508.07901)
[![Project Page](https://img.shields.io/badge/Project_Page-Link-green)](https://www.stand-in.tech)
[![🤗 HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-orange)](https://huggingface.co/BowenXue/Stand-In)

</div>

<img width="5333" height="2983" alt="Image" src="https://github.com/user-attachments/assets/2fe1e505-bcf7-4eb6-8628-f23e70020966" />

> **Stand-In** is a lightweight, plug-and-play framework for identity-preserving video generation. By training only **1%** additional parameters compared to the base video generation model, we achieve state-of-the-art results in both Face Similarity and Naturalness, outperforming various full-parameter training methods. Moreover, **Stand-In** can be seamlessly integrated into other tasks such as subject-driven video generation, pose-controlled video generation, video stylization, and face swapping.

---

## 🔥 News
* **[2025.12.22]** The Stand-In version based on Wan2.2 is now live—feel free to use it! 🚀✨
* **[2025.08.18]** We have released a version compatible with VACE. Not only pose control, but you can also try other control methods such as depth maps, combined with Stand-In to maintain identity simultaneously.

* **[2025.08.16]** We have updated the experimental version of the face swapping feature. Feel free to try it out!

* **[2025.08.13]** Special thanks to @kijai for integrating Stand-In into the custom ComfyUI node **WanVideoWrapper**. However, the implementation differs from the official version, which may affect Stand-In’s performance.  
   In order to address part of the issue, we have urgently released the official Stand-In preprocessing ComfyUI node:  
  👉 https://github.com/WeChatCV/Stand-In_Preprocessor_ComfyUI  
  If you wish to experience Stand-In within ComfyUI, please use **our official preprocessing node** to replace the one implemented by kijai.  
  For the best results, we recommend waiting for the release of our full **official Stand-In ComfyUI**.

* **[2025.08.12]** Released Stand-In v1.0 (153M parameters), the Wan2.1-14B-T2V–adapted weights and inference code are now open-sourced.

---

## 🌟 Showcase

### Identity-Preserving Text-to-Video Generation

| Reference Image | Prompt | Generated Video |
| :---: | :---: | :---: |
|![Image](https://github.com/user-attachments/assets/86ce50d7-8ccb-45bf-9538-aea7f167a541)| "In a corridor where the walls ripple like water, a woman reaches out to touch the flowing surface, causing circles of ripples to spread. The camera moves from a medium shot to a close-up, capturing her curious expression as she sees her distorted reflection." |![Image](https://github.com/user-attachments/assets/c3c80bbf-a1cc-46a1-b47b-1b28bcad34a3) |
|![Image](https://github.com/user-attachments/assets/de10285e-7983-42bb-8534-80ac02210172)| "A young man dressed in traditional attire draws the long sword from his waist and begins to wield it. The blade flashes with light as he moves—his eyes sharp, his actions swift and powerful, with his flowing robes dancing in the wind." |![Image](https://github.com/user-attachments/assets/1532c701-ef01-47be-86da-d33c8c6894ab)|

---
### Non-Human Subjects-Preserving Video Generation

| Reference Image | Prompt | Generated Video |
| :---: | :---: | :---: |
|<img width="415" height="415" alt="Image" src="https://github.com/user-attachments/assets/b929444d-d724-4cf9-b422-be82b380ff78" />|"A chibi-style boy speeding on a skateboard, holding a detective novel in one hand. The background features city streets, with trees, streetlights, and billboards along the roads."|![Image](https://github.com/user-attachments/assets/a7239232-77bc-478b-a0d9-ecc77db97aa5) |

---

### Identity-Preserving Stylized Video Generation

| Reference Image | LoRA | Generated Video |
| :---: | :---: | :---: |
|![Image](https://github.com/user-attachments/assets/9c0687f9-e465-4bc5-bc62-8ac46d5f38b1)|Ghibli LoRA|![Image](https://github.com/user-attachments/assets/c6ca1858-de39-4fff-825a-26e6d04e695f)|
---

### Video Face Swapping

| Reference Video | Identity | Generated Video |
| :---: | :---: | :---: |
|![Image](https://github.com/user-attachments/assets/33370ac7-364a-4f97-8ba9-14e1009cd701)|<img width="415" height="415" alt="Image" src="https://github.com/user-attachments/assets/d2cd8da0-7aa0-4ee4-a61d-b52718c33756" />|![Image](https://github.com/user-attachments/assets/0db8aedd-411f-414a-9227-88f4e4050b50)|


---

### Pose-Guided Video Generation (With VACE)

| Reference Pose | First Frame | Generated Video |
| :---: | :---: | :---: |
|![Image](https://github.com/user-attachments/assets/5df5eec8-b71c-4270-8a78-906a488f9a94)|<img width="719" height="415" alt="Image" src="https://github.com/user-attachments/assets/1c2a69e1-e530-4164-848b-e7ea85a99763" />|![Image](https://github.com/user-attachments/assets/1c8a54da-01d6-43c1-a5fd-cab0c9e32c44)|

---
### For more results, please visit [https://stand-in-video.github.io/](https://www.Stand-In.tech)

## 📖 Key Features
- Efficient Training: Only 1% of the base model parameters need to be trained.
- High Fidelity: Outstanding identity consistency without sacrificing video generation quality.
- Plug-and-Play: Easily integrates into existing T2V (Text-to-Video) models.
- Highly Extensible: Compatible with community models such as LoRA, and supports various downstream video tasks.

---

## ✅ Todo List 
- [x] Release IP2V inference script (compatible with community LoRA).
- [x] Open-source model weights compatible with Wan2.1-14B-T2V: `Stand-In_Wan2.1-T2V-14B_153M_v1.0`。
- [x] Open-source model weights compatible with Wan2.2-T2V-A14B.
- [ ] Release training dataset, data preprocessing scripts, and training code.

---

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Clone the project repository
git clone https://github.com/WeChatCV/Stand-In.git
cd Stand-In

# Create and activate Conda environment
conda create -n Stand-In python=3.11 -y
conda activate Stand-In

# Install dependencies
pip install -r requirements.txt

# (Optional) Install Flash Attention for faster inference
# Note: Make sure your GPU and CUDA version are compatible with Flash Attention
pip install flash-attn --no-build-isolation
```

### 2. Model Download
We provide an automatic download script that will fetch all required model weights into the  `checkpoints` directory.
```bash
python download_models.py
```
This script will download the following models:
* `wan2.1-T2V-14B` (base text-to-video model)
* `antelopev2` (face recognition model)
* `Stand-In` (our Stand-In model)

> Note: If you already have the `wan2.1-T2V-14B model` locally, you can manually edit the `download_model.py` script to comment out the relevant download code and place the model in the `checkpoints/wan2.1-T2V-14B` directory.

---

## 🧪 Usage

### Standard Inference

Use the `infer.py` script for standard identity-preserving text-to-video generation.


```bash
python infer.py \
    --prompt "A man sits comfortably at a desk, facing the camera as if talking to a friend or family member on the screen. His gaze is focused and gentle, with a natural smile. The background is his carefully decorated personal space, with photos and a world map on the wall, conveying a sense of intimate and modern communication." \
    --ip_image "test/input/lecun.jpg" \
    --output "test/output/lecun.mp4"
```
**Prompt Writing Tip:** If you do not wish to alter the subject's facial features, simply use *"a man"* or *"a woman"* without adding extra descriptions of their appearance. Prompts support both Chinese and English input. The prompt is intended for generating frontal, medium-to-close-up videos.

**Input Image Recommendation:** For best results, use a high-resolution frontal face image. There are no restrictions on resolution or file extension, as our built-in preprocessing pipeline will handle them automatically.

---

### Standard Inference - Wan2.2

We provide an automatic download script that will fetch all required model weights into the  `checkpoints` directory.
```bash
python download_models.py --wan_version 2.2
```

```bash
python infer.py \
    --prompt "A man sits comfortably at a desk, facing the camera as if talking to a friend or family member on the screen. His gaze is focused and gentle, with a natural smile. The background is his carefully decorated personal space, with photos and a world map on the wall, conveying a sense of intimate and modern communication." \
    --ip_image "test/input/lecun.jpg" \
    --output "test/output/lecun.mp4" \
    --wan_version "2.2" \
    --fps 16 \
    --base_path checkpoints/Wan2.2/t2v
```
**Prompt Writing Tip:** If you do not wish to alter the subject's facial features, simply use *"a man"* or *"a woman"* without adding extra descriptions of their appearance. Prompts support both Chinese and English input. The prompt is intended for generating frontal, medium-to-close-up videos.

**Input Image Recommendation:** For best results, use a high-resolution frontal face image. There are no restrictions on resolution or file extension, as our built-in preprocessing pipeline will handle them automatically.

---

### Inference with Community LoRA

Use the `infer_with_lora.py` script to load one or more community LoRA models alongside Stand-In.

```bash
python infer_with_lora.py \
    --prompt "A man sits comfortably at a desk, facing the camera as if talking to a friend or family member on the screen. His gaze is focused and gentle, with a natural smile. The background is his carefully decorated personal space, with photos and a world map on the wall, conveying a sense of intimate and modern communication." \
    --ip_image "test/input/lecun.jpg" \
    --output "test/output/lecun.mp4" \
    --lora_path "path/to/your/lora.safetensors" \
    --lora_scale 1.0
```

We recommend using this stylization LoRA: [https://civitai.com/models/1404755/studio-ghibli-wan21-t2v-14b](https://civitai.com/models/1404755/studio-ghibli-wan21-t2v-14b)

---

### Video Face Swapping

Use the `infer_face_swap.py` script to perform video face swapping with Stand-In.

```bash
python infer_face_swap.py \
    --prompt "The video features a woman standing in front of a large screen displaying the words ""Tech Minute"" and the logo for CNET. She is wearing a purple top and appears to be presenting or speaking about technology-related topics. The background includes a cityscape with tall buildings, suggesting an urban setting. The woman seems to be engaged in a discussion or providing information on technology news or trends. The overall atmosphere is professional and informative, likely aimed at educating viewers about the latest developments in the tech industry." \
    --ip_image "test/input/ruonan.jpg" \
    --output "test/output/ruonan.mp4" \
    --denoising_strength 0.85
```
**Note**: Since Wan2.1 itself does not have an inpainting function, our face swapping feature is still experimental.

The higher the denoising_strength, the more the background area is redrawn, and the more natural the face area becomes. Conversely, the lower the denoising_strength, the less the background area is redrawn, and the higher the degree of overfitting in the face area.

You can set --force_background_consistency to make the background completely consistent, but this may lead to potential and noticeable contour issues. Enabling this feature requires experimenting with different denoising_strength values to achieve the most natural effect. If slight changes to the background are not a concern, please do not enable this feature.


### Infer with VACE
Use the `infer_with_vace.py` script to perform identity-preserving video generation with Stand-In, compatible with VACE.
```bash
python infer_with_vace.py \
    --prompt "A woman raises her hands." \
    --vace_path "checkpoints/VACE/" \
    --ip_image "test/input/first_frame.png" \
    --reference_video "test/input/pose.mp4" \
    --reference_image "test/input/first_frame.png" \
    --output "test/output/woman.mp4" \
    --vace_scale 0.8
```
You need to download the corresponding weights from the `VACE` repository or provide the path to the `VACE` weights in the `vace_path` parameter.

```bash
python download_models.py --vace
```

The input control video needs to be preprocessed using VACE's preprocessing tool. Both `reference_video` and `reference_image` are optional and can exist simultaneously. Additionally, VACE’s control has a preset bias towards faces, which affects identity preservation. Please lower the `vace_scale` to a balance point where both motion and identity are preserved. When only `ip_image` and `reference_video` are provided, the weight can be reduced to 0.5.

Using both Stand-In and VACE together is more challenging than using Stand-In alone. We are still maintaining this feature, so if you encounter unexpected outputs or have other questions, feel free to raise them in the issue.


## 🤝 Acknowledgements

This project is built upon the following excellent open-source projects:
* [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) (training/inference framework)
* [Wan2.1](https://github.com/Wan-Video/Wan2.1) (base video generation model)

We sincerely thank the authors and contributors of these projects.

The original raw material of our dataset was collected with the help of our team member [Binxin Yang](https://binxinyang.github.io/), and we appreciate his contribution!

---

## ✏ Citation

If you find our work helpful for your research, please consider citing our paper:

```bibtex
@article{xue2025standin,
      title={Stand-In: A Lightweight and Plug-and-Play Identity Control for Video Generation}, 
      author={Bowen Xue and Qixin Yan and Wenjing Wang and Hao Liu and Chen Li},
      journal={arXiv preprint arXiv:2508.07901},
      year={2025},
}
```

---

## 📬 Contact Us

If you have any questions or suggestions, feel free to reach out via [GitHub Issues](https://github.com/WeChatCV/Stand-In/issues) . We look forward to your feedback!
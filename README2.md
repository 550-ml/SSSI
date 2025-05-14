# SSSI: Self-Prompted Segmentation of Scientific Illustrations

This repository contains the official implementation of the paper:  
**"SSSI: Self-Prompted Segmentation of Scientific Illustrations"**, submitted to [ICDAR2025].

> 📌 This project focuses on segmenting semantically meaningful subregions from complex scientific flowcharts, enabling better visual-text alignment in downstream tasks.
It is particularly motivated by the need for automatically generating AI-assisted presentation slides for scientific papers, where decomposing illustrations allows finer-grained alignment between figure content and textual descriptions in the paper.


---

## 🔍 Overview
![SSSI architecture](assets/framework.png?raw=true)
The SSSI framework leverages the textual layout and pre-segmentation cues in scientific flowcharts to generate multi-type prompts for SAM, including points, bounding boxes, and masks.  
Through dynamic point sampling and iterative refinement, SSSI is able to segment semantically complete subregions such as stages, encoders, or input/output modules — rather than over-fragmented parts.

![SSSI result](assets/result.png?raw=true)
**Figure:** (a) Visualization of segmentation results by **SSSI**. Each subregion corresponds to a semantically complete unit (e.g., "Stage", "Encoder") and is clearly delineated.  
(b) Result from the original **SAM**.

---

## 📁 Dataset: SciFigSeg

We introduce a new dataset, **SciFigSeg**, consisting of 723 manually annotated scientific illustrations collected from arXiv. Each subregion is annotated as a polygon in COCO format.

Download instructions:  
👉 [Link to dataset release or `dataset/README.md`](./dataset/README.md)

---

## 🛠 Installation
We recommend using **Python 3.12** and **PyTorch ≥ 2.5.0**
```bash
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install paddlepaddle-gpu==2.6.2 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
pip install -r requirements.txt
pip install -e .
```

## 🚀 Usage

### 📦 Prepare Checkpoints and Input

Run the script to download the checkpoint, and place your input images under `dataset/images/`.

### 🔍 Segment Illustrations

Run SSSI on image folder to generate subregion masks and visualize the results:

```bash
cd Inference
python inference_image_folder.py
```

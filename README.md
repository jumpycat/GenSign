<div align="center">

## Enabling Supervised Learning of Generative Signatures for Generalized AI-Generated Images Detection

**CVPR 2026**

**Jianwei Fei\***, Yunshu Dai, Xiaoyu Zhou, Zhihua Xia, Alessandro Piva

University of Florence · Sun Yat-sen University · Jinan University

</div>

---

## Repository Status

### Available
- [x] Training code
- [x] Pretrained checkpoints
- [x] Additional documentation
- [x] Evaluation code

---



## 1. DAIR Simulator

The Dynamic-Architecture Image Reconstructor (DAIR) used in this project is released as an independent repository:

https://github.com/jumpycat/DAIR

Please refer to the DAIR repository for details regarding the simulator implementation.



## 2. GenSign Extractor

`gensign_extractor` is a lightweight fully convolutional network for directly extracting (predicting) a hidden noise pattern. The model is based on `DenoisingFCNWithSkip`.

### Files

- **`train_gensign.py`** Training script. Uses the pretrained DAIR to compute `gt_noise = imgs - recon_imgs` as the supervision signal, and trains `gensign_extractor` (referred to as `noiser` in the script) to fit this noise.
- **`gensign_extractor.py`** Model definition (`DenoisingFCNWithSkip`), i.e., the network architecture of the signature extractor.
- **`extract_gensign.py`** Inference script. Loads the trained extractor weights, reads all images in a given folder, runs forward inference on each, takes the fractional part of the output as the final extracted signature noise, and saves it as an image.
- **`Pretrained Weights`** [gensign_extractor weights (Google Drive)](https://drive.google.com/file/d/1u586GP5nXksUTJvvd69xBu6lt4QmqMnn/view?usp=drive_link). After downloading, place it anywhere and pass its path via `--checkpoint`.

### Usage

**Training**

```bash
python train_gensign.py \
    --train_dir /path/to/train_images \
    --resume /path/to/encoder_decoder_checkpoint.pth
```

**Extracting Signature Noise**
```bash
python extract_gensign.py \
    --input_dir /path/to/images \
    --output_dir /path/to/output \
    --checkpoint /path/to/gensign_extractor.pth
```
The extracted results are saved in `output_dir` as `<original_filename>_noise.png`, with one extracted noise image per input image.


## 3. The Dual Detector
### Pretrained Checkpoints

We provide checkpoints trained on different datasets:

| Training Data | Download |
|---|---|
| GenImage-SD1.4 | [Link](https://drive.google.com/file/d/13_gaGNLMXoiA4Wbg8IHFwPBtMHm4zloK/view?usp=drive_link) |
| ProGAN (4-class) | [Link](https://drive.google.com/file/d/18unjmMmRUsGDhcYx8OyctZL0ytMR2wjZ/view?usp=drive_link) |
| ProGAN (20-class) | [Link](https://drive.google.com/file/d/1zRL5934GDEnFtUUuSL59RukKjC_Pwfab/view?usp=drive_link) |


### Overview

The Dual Detector is the core classification framework that fuses semantic features and generative signature features to detect AI-generated images. It employs a two-stream architecture:
1. **Stream 1 (Semantic):** A pre-trained CLIP vision model fine-tuned using LoRA.
2. **Stream 2 (Signature):** A frozen GenSign Extractor (`noiser`) coupled with an EfficientNet backbone to process generative noise residuals.

During training, the model dynamically leverages the DAIR Simulator to perform targeted data augmentation, improving generalization across different generative models.

### Code Structure

- **`dual_model.py`**: Defines the `DualStreamModel` architecture, LoRA injection, and the feature fusion head.
- **`datasets.py`**: Modularized dataset classes and loaders handling real/fake label mapping and transformations.
- **`eval_utils.py` & `utils.py`**: Modular evaluation logic for computing Accuracy, AUC, and AP, alongside utility functions.
- **`train_genimg.py`**: Training pipeline tailored for the GenImage dataset structure.
- **`train_progan.py`**: Training pipeline tailored for the ProGAN dataset structure (e.g., CNNDetection) with multi-dataset evaluation.

### Training

To train the Dual Detector, you must provide the pre-trained checkpoints for both the DAIR simulator (for data augmentation) and the GenSign Extractor (as the frozen feature extractor for Stream 2).

**Training on GenImage:**

```
python train_genimg.py \
    --train_dir /path/to/GenImage/train \
    --val_dir /path/to/GenImage/val \
    --eval_datasets_root /path/to/GenImage/valdata \
    --ae_resume /path/to/dair_simulator.pth \
    --noiser_resume /path/to/gensign_extractor.pth \
    --stream1_model openai/clip-vit-large-patch14 \
    --stream2_model efficientnet-b0 \
    --aug_target all \
    --aug_prob 0.2 \
    --batch_size 8
```

```
python train_progan.py \
    --train_dir /path/to/progan_train \
    --train_subfolders car cat chair horse \
    --ae_resume /path/to/dair_simulator.pth \
    --noiser_resume /path/to/gensign_extractor.pth \
    --stream1_model openai/clip-vit-large-patch14 \
    --stream2_model efficientnet-b0 \
    --aug_target all \
    --aug_prob 0.2 \
    --batch_size 8
``` 

### Notes

- Using `1`/`0` as the label for fake images can lead to asymmetric results. We recommend using `1` as the label for **real** images.
- Using the default threshold of `0.5` may produce results inconsistent with AUC and AP. We mainly report **AP** and **AUC**; computing accuracy (Acc) may require additional threshold calibration, as `0.5` may not be the optimal choice.



---

## Citation

```bibtex
@inproceedings{fei2026enabling,
  title={Enabling Supervised Learning of Generative Signatures for Generalized AI-Generated Images Detection},
  author={Fei, Jianwei and Dai, Yunshu and Zhou, Xiaoyu and Xia, Zhihua and Piva, Alessandro},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={14084--14094},
  year={2026}
}
```

---

## Contact

Jianwei Fei: fei_jianwei@163.com

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

### Coming Soon

- [ ] Evaluation code
- [ ] Pretrained checkpoints
- [ ] Additional documentation

---

## DAIR Simulator

The Dynamic-Architecture Image Reconstructor (DAIR) used in this project is released as an independent repository:

https://github.com/jumpycat/DAIR

Please refer to the DAIR repository for details regarding the simulator implementation.

## GenSign Extractor

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
```

# Trap-MID: Trapdoor-based Defense against Model Inversion Attacks

This is the official PyTorch implementation of the paper **Trap-MID: Trapdoor-based Defense against Model Inversion Attacks**.

> Model Inversion (MI) attacks pose a significant threat to the privacy of Deep Neural Networks by recovering training data distribution from well-trained models. While existing defenses often rely on regularization techniques to reduce information leakage, they remain vulnerable to recent attacks. In this paper, we propose the **Trap**door-based **M**odel **I**nversion **D**efense (Trap-MID) to mislead MI attacks. A trapdoor is integrated into the model to predict a specific label when the input is injected with the corresponding trigger. Consequently, this trapdoor information serves as the "shortcut" for MI attacks, leading them to extract trapdoor triggers rather than private data. We provide theoretical insights into the impacts of trapdoor's effectiveness and naturalness on deceiving MI attacks. In addition, empirical experiments demonstrate the state-of-the-art defense performance of Trap-MID against various MI attacks without the requirements for extra data or large computational overhead.

Full Paper: https://arxiv.org/abs/2411.08460

## Requirements

This code was tested with `Python 3.9.16`, `PyTorch 2.1.1` and `CUDA 11.8`.

Install Pytorch according to your CUDA version. For example:

```bash
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
```

Install other dependencies:

```bash
pip install -r requirements.txt
```

## Preparation

- Download [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset.
- For FaceNet-64 and ResNet-152 models, the pre-trained checkpoints can be downloaded at https://github.com/SCccc21/Knowledge-Enriched-DMI.

## Model Training

Modify `./config/classify_trap.json` to configure model training and execute the following script:

```bash
python train.py \
    --config ./config/classify_trap.json
```

## Model Inversion Attacks

Directly load the checkpoints and conduct Model Inversion Attacks with their official codes:

- GMI: https://github.com/MKariya1998/GMI-Attack
- KED-MI: https://github.com/SCccc21/Knowledge-Enriched-DMI
- LOMMA: https://github.com/sutd-visual-computing-group/Re-thinking_MI
- PLG-MI: https://github.com/LetheSec/PLG-MI-Attack

or attack with the implementation in `./plgmi` or `./lomma`:

- For GMI and KED-MI, execute the code in `./plgmi/baselines`.
- For PLG-MI, execute the code in `./plgmi`.
- For LOMMA, execute the code in `./lomma`.

## Reference

The codes for model training are mainly modified from https://github.com/SCccc21/Knowledge-Enriched-DMI. We adopt the implementations in https://github.com/sutd-visual-computing-group/Re-thinking_MI and https://github.com/LetheSec/PLG-MI-Attack for MI attacks.

1. Yuheng Zhang, Ruoxi Jia, Hengzhi Pei, Wenxiao Wang, Bo Li, and Dawn Song. The secret revealer: Generative model-inversion attacks against deep neural networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 253–261, 2020.
2. Si Chen, Mostafa Kahla, Ruoxi Jia, and Guo-Jun Qi. Knowledge-enriched distributional model inversion attacks. In Proceedings of the IEEE/CVF international conference on computer vision, pages 16178–16187, 2021.
3. Ngoc-Bao Nguyen, Keshigeyan Chandrasegaran, Milad Abdollahzadeh, and Ngai-Man Cheung. Re-thinking model inversion attacks against deep neural networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16384–16393, 2023.
4. Xiaojian Yuan, Kejiang Chen, Jie Zhang, Weiming Zhang, Nenghai Yu, and Yang Zhang.325
Pseudo label-guided model inversion attack via conditional generative adversarial network.326
AAAI 2023, 2023.

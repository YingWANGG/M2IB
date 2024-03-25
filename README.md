# Visual Explanations of Image–Text Representations via Multi-Modal Information Bottleneck Attribution

This repository contains the official implementation for


**Visual Explanations of Image-Text Representations via Multi-Modal Information Bottleneck Attribution**.<br>
[Ying Wang](https://yingwangg.github.io/)\*, [Tim G. J. Rudner](https://timrudner.com/)\*, [Andrew Gordon Wilson](https://cims.nyu.edu/~andrewgw/). **NeurIPS 2023**. (\* indicates equal contribution)

**Abstract:** Vision-language pretrained models have seen remarkable success, but their application to safety-critical settings is limited by their lack of interpretability. To improve the interpretability of vision-language models such as CLIP, we propose a multi-modal information bottleneck (M2IB) approach that learns latent representations that compress irrelevant information while preserving relevant visual and textual features. We demonstrate how M2IB can be applied to attribution analysis of vision-language pretrained models, increasing attribution accuracy and improving the interpretability of such models when applied to safety-critical domains such as healthcare. Crucially, unlike commonly used unimodal attribution methods, M2IB does not require ground truth labels, making it possible to audit representations of vision-language pretrained models when multiple modalities but no ground-truth data is available. Using CLIP as an example, we demonstrate the effectiveness of M2IB attribution and show that it outperforms gradient-based, perturbation-based, and attention-based attribution methods both qualitatively and quantitatively.

<p align="center">
  &#151; <a href="https://openreview.net/forum?id=ECvtxmVP0x"><b>View Paper</b></a> &#151;
</p>

<br>

![](https://github.com/YingWANGG/M2IB/blob/main/images/visualization.png)

### Quick start
[Colab](https://colab.research.google.com/drive/1TeRDHYg4AXbQf0XqUcgC-3sH0fJe4Tia?usp=sharing)

[Jupyter Notebook](https://github.com/YingWANGG/M2IB/blob/main/demo.ipynb)

```
python scripts/eval.py
--data_path <your data path, containing image paths and corresponding captions>
--output_path <your output path that will store the evaluation results>
--samples <number of samples randomly drawn from the image-text pairs in data_path>
```

### Citation
If you found our paper or code useful, please cite it as:
```
@inproceedings{
  wang2023visual,
  title={Visual Explanations of Image-Text Representations via Multi-Modal Information Bottleneck Attribution},
  author={Ying Wang and Tim G. J. Rudner and Andrew Gordon Wilson},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023},
  url={https://openreview.net/forum?id=ECvtxmVP0x}
}
```

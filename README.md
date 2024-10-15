# SlideGCD

This repository provides the Pytorch implementations of the paper titled "SlideGCD: Slide-based Graph Collaborative 
Training with Knowledge Distillation for Whole Slide Image Classification" and accepted by Medical Image Computing 
and Computer Assisted Intervention (MICCAI), 2024. The paper has been officially published at [SpringerLink](https://link.springer.com/chapter/10.1007/978-3-031-72083-3_44) and is also available at [Arxiv](https://arxiv.org/abs/2407.08968).

## News

**The extensive vision of this conference paper is now preprinted at [ArXiv](https://arxiv.org/abs/2410.10260), and its related source code will be released soon.**

**A new interactable visualization of the Slide-based Graph is being planned and will be released soon.**

## Download the WSIs

We provide the slide list and dataset partition used to evaluate our methods in ./data.

The WSIs can be found in the TCGA project:

https://www.cancer.gov/about-nci/organization/ccg/research/structural-genomics/tcga

## Patch Extraction

We directly use the pre-trained PLIP (Pathology Language and Image Pre-Training) 
as our Patch Encoder. You can follow the official repository [here](https://github.com/PathologyFoundation/plip) 
for patch extraction yourselves.

## Citation

Please cite this work if you consider it useful via
```
Shu, T., Shi, J., Sun, D., Jiang, Z., Zheng, Y. (2024). SlideGCD: Slide-Based Graph Collaborative Training with Knowledge Distillation for Whole Slide Image Classification. In: Linguraru, M.G., et al. Medical Image Computing and Computer Assisted Intervention – MICCAI 2024. MICCAI 2024. Lecture Notes in Computer Science, vol 15004. Springer, Cham. https://doi.org/10.1007/978-3-031-72083-3_44
```
or
```
Shu T, Shi J, Sun D, et al. SlideGCD: Slide-based Graph Collaborative Training with Knowledge Distillation for Whole 
Slide Image Classification[J]. arXiv preprint arXiv:2407.08968, 2024.
```
and we will be very pleased.

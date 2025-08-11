# SlideGCD

This repository provides the Pytorch implementations of the papers titled "SlideGCD: Slide-based Graph Collaborative 
Training with Knowledge Distillation for Whole Slide Image Classification" (accepted by Medical Image Computing 
and Computer Assisted Intervention (MICCAI), 2024) and "Slide-based Graph Collaborative Training for Histopathology Whole Slide Image Analysis" (accepted by IEEE Transactions on Medical Imaging).

## News

**19/05/2025 The extensive journal paper titled "Slide-based Graph Collaborative Training for Histopathology Whole Slide Image Analysis" has been accepted by IEEE Transactions on Medical Imaging [10.1109/TMI.2025.3571152](https://ieeexplore.ieee.org/abstract/document/11007022)!**

**09/04/2025 The related source code of the extensive version manuscript is now released, indicated as SlideGCDv2.**

**14/10/2024 The extensive version of this conference paper is now preprinted at [ArXiv](https://arxiv.org/abs/2410.10260), and its related source code will be released soon.**

<!--**14/10/2024 A new interactable visualization of the Slide-based Graph is being planned and will be released soon.**-->

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
J. Shi, T. Shu, Z. Jiang, W. Wang, H. Wu and Y. Zheng, "Slide-based Graph Collaborative Training for Histopathology Whole Slide Image Analysis," in IEEE Transactions on Medical Imaging, doi: 10.1109/TMI.2025.3571152.
```
and
```
Shu, T., Shi, J., Sun, D., Jiang, Z., Zheng, Y. (2024). SlideGCD: Slide-Based Graph Collaborative Training with Knowledge Distillation for Whole Slide Image Classification. In: Linguraru, M.G., et al. Medical Image Computing and Computer Assisted Intervention – MICCAI 2024. MICCAI 2024. Lecture Notes in Computer Science, vol 15004. Springer, Cham. https://doi.org/10.1007/978-3-031-72083-3_44
```
and we will be very pleased.

---
license: mit
task_categories:
- image-classification
size_categories:
- 10K<n<100K
---

## EuroSAT-SAR: Land Use and Land Cover Classification with Sentinel-1

The EuroSAT-SAR dataset is a SAR version of the popular [EuroSAT](https://github.com/phelber/EuroSAT) dataset. We matched each Sentinel-2 image in EuroSAT with one Sentinel-1 patch according to the geospatial coordinates, ending up with 27,000 dual-pol Sentinel-1 SAR images divided in 10 classes. The EuroSAT-SAR dataset was collected as one downstream task in the work [FG-MAE](https://github.com/zhu-xlab/FGMAE) to serve as a CIFAR-like, clean, balanced ML-ready dataset for remote sensing SAR image recognition.

<p align="center">
  <img width="1000" alt="fgmae main structure" src="assets/eurosat-sar.png">
</p>

The dataset can be downloaded as a compressed zip file [here](https://huggingface.co/datasets/wangyi111/EuroSAT-SAR/resolve/main/EuroSAT-SAR.zip).

### Citation
```bibtex
@article{wang2023feature,
  title={Feature Guided Masked Autoencoder for Self-supervised Learning in Remote Sensing},
  author={Wang, Yi and Hern{\'a}ndez, Hugo Hern{\'a}ndez and Albrecht, Conrad M and Zhu, Xiao Xiang},
  journal={arXiv preprint arXiv:2310.18653},
  year={2023}
}
```
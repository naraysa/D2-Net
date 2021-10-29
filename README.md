# D2-Net
[ICCV 2021] D2-Net: Weakly-Supervised Action Localization via Discriminative Embeddings and Denoised Activations. [Arxiv link](https://arxiv.org/abs/2012.06440)

## Training D2-Net
Train on Thumos14 dataset using the following command.
```
python main.py --model-name Th14_release --dataset-name Thumos14reduced --num-class 20 --summary "D2net" --cuda
```

## Citation
Kindly cite the following work if you find this repo useful.
```
@inproceedings{narayan2021d2,
  title={D2-Net: Weakly-supervised action localization via discriminative embeddings and denoised activations},
  author={Narayan, Sanath and Cholakkal, Hisham and Hayat, Munawar and Khan, Fahad Shahbaz and Yang, Ming-Hsuan and Shao, Ling},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2021}
}
```

## Dependencies
This codebase has been tested on PyTorch1.7 and was based on the W-TALC repo found [here](https://github.com/sujoyp/wtalc-pytorch).

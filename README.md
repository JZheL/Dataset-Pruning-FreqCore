# 📍 Dataset-Pruning-FreqCore
Official PyTorch implementation of paper (SIGKDD 2026) 🤩

"FreqCore: A Frequency Domain Perspective on Coreset Selection" 
>[Jiazhe Li](https://github.com/JZheL), [Chenhe Hao](https://github.com/xrosssaber12306), [Weiying Xie](https://scholar.google.com/citations?user=y0ha5lMAAAAJ&hl=zh-CN), [Jitao Ma](https://orcid.org/0009-0009-7782-8184), [Daixun Li](https://scholar.google.cz/citations?user=gaiP4-IAAAAJ&hl=zh-CN&oi=ao), [Xin Zhang](https://scholar.google.com/citations?user=quAaEpgAAAAJ&hl=zh-CN), [Leyuan Fang](https://scholar.google.cz/citations?user=Gfa4nasAAAAJ&hl=zh-CN&oi=ao)<br>
>XDU and HNU
![image](https://github.com/JZheL/Dataset-Pruning-FreqCore/blob/master/framework.png)

## 📝 Abstract
Coreset selection aims to select an informative subset that achieves performance comparable to the whole dataset. However, existing methods are often susceptible to spurious correlations present in high-frequency domain, consequently leading to inaccurate sample importance estimation and suboptimal coresets. To address the problem, we propose FreqCore, a novel dynamic coreset selection method that focuses on the robust Low-Frequency Property (LFP) of hidden representations. Building upon stable low-frequency components, FreqCore inherently mitigates spurious high-frequency correlations, enabling a more accurate estimation of sample importance. Our FreqCore is directly inspired by the key discovery that the Global Commonality Subspace (GCS) projection of LFP effectively reveals Information Redundancy (IR). Based on the observation, we mathematically formulate a criterion that prunes high-IR samples, as they make minimal contribution to dataset diversity, and retains low-IR samples that capture unique information. Without introducing additional constraints, FreqCore significantly reduces computational costs. Extensive experiments demonstrate the superior generalization capability of FreqCore. For example, with ResNet-18 on CIFAR-100 at a 70\% pruning rate, FreqCore achieves 79.08\% accuracy, surpassing the full dataset by 0.88\%. The excellent performance also extends to Transformer architectures. With ViT-Ti-16 on CIFAR-100 with a 50\% pruning ratio, FreqCore achieves 81.28\% accuracy, outperforming the full dataset by 0.44\%.


## Requirements 🌏
Python >= 3.8
  

## Experiments 🏃🏻‍♀️
Remember to change the dataset path! And you can change the pruning rate by adjusting `--ratrio`.
### CIFAR10 with ResNet18
* `python examples/cifar_r18.py --use_freqcore --dataset_name cifar10 --save_path /Path/to/save/checkpoints/and/log --ratio 0.7`

### CIFAR100 with ResNet18
* `python examples/cifar_r18.py --use_freqcore --dataset_name cifar100 --save_path /Path/to/save/checkpoints/and/log --ratio 0.7`

### CIFAR100 with ViT-Ti-16
We use a pre-trained model, whose weight parameters are located in `examples/vit_ti_16.bin`.

* `python examples/cifar_vit.py --use_freqcore --pretrained-path --resize-224 examples/vit_ti_16.bin --save_path path/to/save/checkpoints/and/log --ratio 0.7`

### Resume the Interrupted Training
If your training is interrupted, don't worry, our code supports resuming training from a checkpoint.

* `python examples/cifar_r18.py --use_freqcore --dataset_name cifar10 --save_path /Path/to/save/checkpoints/and/log --ratio 0.7 --resume path/to/latest/checkpoint --start_epoch X --manualSeed X`

Explanation:

Note 1: `--start_epoch X` means resuming training from the latest epoch. Thus, you need to replace `X` with the epoch index of the saved latest checkpoint path.

Note 2: `--manualSeed X` means that the seed for the resumed training and the training before the interruption must be the same. Thus, you need to replace `X` with the seed from the training before the interruption.

Example：
Assuming training was interrupted before epoch 71 was completed, and the results are saved in `./result`, currently containing `checkpoint_epoch_70.pth` and `log_seed_1071.txt`, the command to resume training is:
* `python examples/cifar_r18.py --use_freqcore --dataset_name cifar10 --save_path ./result --ratio 0.7 --resume result/checkpoint_epoch_70.pth --start_epoch 70 --manualSeed 1071`

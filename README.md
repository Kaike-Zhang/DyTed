# DyTed: Disentangled Representation Learning for Discrete-time Dynamic Graph

**Authors**: 
- **Kaike Zhang**
- Qi Cao
- Gaolin Fang
- Bingbing Xu
- Hongjian Zou
- Huawei Shen
- Xueqi Cheng

## Abstract
Unsupervised representation learning for dynamic graphs has attracted a lot of research attention in recent years. Compared with static graph, the dynamic graph is a comprehensive embodiment of both the intrinsic stable characteristics of nodes and the time-related dynamic preference. However, existing methods generally mix these two types of information into a single representation space, which may lead to poor explanation, less robustness, and a limited ability when applied to different downstream tasks. To solve the above problems, in this paper, we propose a novel disenTangled representation learning framework for discrete-time Dynamic graphs, namely DyTed. We specially design a temporal-clips contrastive learning task together with a structure contrastive learning to effectively identify the time-invariant and time-varying representations respectively. To further enhance the disentanglement of these two types of representation, we propose a disentanglement-aware discriminator under an adversarial learning framework from the perspective of information theory. Extensive experiments on Tencent and five commonly used public datasets demonstrate that DyTed, as a general framework that can be applied to existing methods, achieves state-of-the-art performance on various downstream tasks, as well as be more robust against noise.
[Link to the paper on ACM Digital Library](https://dl.acm.org/doi/abs/10.1145/3580305.3599319)


## Environment
- networkx >=2.6.3
- numpy >= 1.22.2
- scikit-learn >= 1.0.2
- scipy >= 1.8.0
- torch >= 1.9.1
- torch-cluster >= 1.5.9
- torch-geometric >= 1.7.2
- torch-scatter >= 2.0.8
- torch-sparse >= 0.6.11
- torch-spline-conv >= 1.2.1


## Usage (Quick Start)
1. Install the required packages using pip:

    ```bash
    pip install -r requirements.txt
    ```

2. Run the main script with the desired backbone model and dataset:

    ```bash
    python main.py --model=<backbone model> --dataset=<dataset>
    ```

   Replace `<backbone model>` with the name of your model, and `<dataset>` with the name of your dataset.

3. If you need to use more backbone models, make sure the model's input format is converted to a list compatible with `torch_geometric.data`. For converting sparse matrices to `torch_geometric.data`, refer to the `_build_pyg_graphs()` function inside the `./utilize/trainer.py` file. The output should be a list of representations.



## Citation
If you find our work useful, please cite our paper using the following BibTeX:

```bibtex
@inproceedings{zhang2023dyted,
  title={DyTed: Disentangled Representation Learning for Discrete-time Dynamic Graph},
  author={Zhang, Kaike and Cao, Qi and Fang, Gaolin and Xu, Bingbing and Zou, Hongjian and Shen, Huawei and Cheng, Xueqi},
  booktitle={Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={3309--3320},
  year={2023}
}

# Divide-and-Distill MIL : Expert Clustering and Knowledge Transfer for Whole Slide Image Classification

## Abstract
Multiple Instance Learning (MIL) is an effective paradigm for Whole Slide Image (WSI) classification. Despite its success, existing MIL methods often suffer from limited learning capacity and struggle to fully exploit the rich information within WSIs, leading to performance plateaus. To overcome these challenges and improve the performance of any existing MIL method without additional overhead at inference, we propose the Divide-and-Distill (D&D) framework, which augments MIL methods through a two-stage process: feature space partitioning via clustering and knowledge transfer through distillation. By training specialized experts on localized clusters of the feature space and distilling their knowledge into a single final model, D&D enables both global and localized feature learning, pushing the boundaries of MIL methods. 

## Python Virtual Environment Setup
- **Python Version**: 3.10
- **CUDA Version**: 11.8.0 (for GPU support)

 ### Virtual Environment Creation
```bash
conda create -n dndmil python=3.10 -y
conda activate dnd
```
### Key package versions

- `torch` 2.0.1
- `torchvision` 0.15.2
- `torchaudio` 2.0.2
- `einops` 0.8.0 
- `k-means-constrained` 0.7.3
- `scikit-learn` 1.3.2
-  `causal-conv1d` 1.1.1
-   `mamba-ssm` 1.1.2

## How to Run the Code
### Data Preparation
1. Download Camelyon16, TCGA NSCLC (LUAD & LUSC), and BRACS WSIs.
2. Use the CLAM framework to extract features at 10x resolution
3. Organize the extracted .pt feature files as follows:

```bash
ROOT_DIR/
    └──${FEATURE_EXTRACTOR_NAME}/
            └──${DATASET_NAME}_features/
                    ├── slide_1.pt
                    ├── slide_2.pt
                    └── ...
```

## Acknowledgements
Many thanks to the authors of  [CLAM](https://github.com/mahmoodlab/CLAM) and [MambaMIL](https://github.com/isyangshu/MambaMIL) for making their codebase open-source and accessible to other researchers.

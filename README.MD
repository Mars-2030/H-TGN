
# Beyond the Next Link: A System for Discovering and Visualizing Evolving Community Structure in Dynamic Graphs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation for the paper: **"Beyond the Next Link: A System for Discovering and Visualizing Evolving Community Structure in Dynamic Graphs"**.

This repository contains the source code and experimental configurations needed to reproduce the results presented in our work.

***
> **Note:** The accompanying paper has been submitted to the **ACM SIGMOD 2026 International Conference on Management of Data** and is currently under review.
***
---

## Table of Contents
- [About the Project](#about-the-project)
- [Installation](#installation)
- [Running the Experiments](#running-the-experiments)
  - [Training a New Model](#training-a-new-model)
  - [Reproducing Paper Results](#reproducing-paper-results)
- [Generating Analytical Insights](#generating-analytical-insights)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## About the Project

This project introduces a novel analytical system for understanding the meso-scale dynamics of continuous-time temporal graphs. While state-of-the-art models like Temporal Graph Networks (TGNs) excel at predicting the "next link," they provide little insight into *why* these links occur or what high-level structures govern them.

Our system addresses this gap by temporalizing the established SRC (Selection, Reduction, Connection) framework for graph pooling. By integrating a temporal, hierarchical coarsening module into a TGN backbone, our system learns to group nodes into evolving communities, generating an interpretable, multi-scale summary of the network's dynamics.

> **A Note on Our Core Finding:** Our rigorous experiments led to a crucial and counter-intuitive insight. We found that the learned hierarchy, while computationally expensive, offers **no significant predictive advantage** over a strong TGN baseline for link prediction. Instead, its true and unique value lies in its ability to serve as a powerful **analytical lens**. This repository provides the tools to use our system not as a SOTA predictor, but as a novel instrument for discovering and interpreting the hidden community structures that evolve within dynamic networks.

## Installation

This project is built using PyTorch and PyTorch Geometric. We recommend using a Conda environment to manage dependencies.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/temporal-graph-coarsening.git
    cd temporal-graph-coarsening
    ```

2.  **Create and activate the Conda environment:**
    All required packages are listed in `environment.yml`. This includes specific CUDA versions for PyTorch, so please adjust if necessary.
    ```bash
    conda env create -f environment.yml
    conda activate temporal-coarsening
    ```

3.  **Download Datasets:**
    The JODIE datasets (Reddit, Wikipedia, MOOC, LastFM) and the ICEWS18 dataset will be automatically downloaded by the PyTorch Geometric loaders upon first run and saved to a `data/` directory.

## Running the Experiments

The main training scripts are `icews18.py` and `reddit.py`.

### Training a New Model

You can train a model from scratch by running the corresponding script with your desired hyperparameters.

**Example: Training on ICEWS18**
```bash
python icews18.py \
    --seed 123453 \
    --static_loss_weight 0.8 \
    --lr 0.0002
```

**Example: Training on Reddit**
```bash
python reddit.py \
    --seed 123452 \
    --static_loss_weight 0.8 \
    --lr 0.0002
```



## Generating Analytical Insights

The primary output of our system is the set of visualizations that reveal community dynamics. After a model is trained, it will automatically save a series of plots and reports to the project's root directory, including:

-   **Community Evolution Alluvial/Sankey Diagrams:** Trace the merging, splitting, and flow of nodes between communities over time.
-   **Hierarchical t-SNE Plots:** Visualize the emergence of a clustered structure in the embedding space.
-   **Cluster Utilization Heatmaps:** Show the "lifecycles" of communities, indicating when they are active or dormant.
-   **Hierarchical Analysis Reports:** Provide a textual summary of the coarsened graph and the composition of the largest communities.


## Acknowledgments
The authors gratefully acknowledge the use of Gemini by Google as a writing assistant in the preparation of the associated manuscript. Gemini was utilized for tasks including refining prose, improving the clarity of arguments, providing assistance with LaTeX formatting, and debugging portions of the implementation code.




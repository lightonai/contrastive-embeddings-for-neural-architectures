# <img src="_static/lighton_small.png" width=60/> Contrastive Embeddings for Neural Architectures

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)  [![Twitter](https://img.shields.io/twitter/follow/LightOnIO?style=social)](https://twitter.com/LightOnIO)

The performance of algorithms for neural architecture search strongly depends on the parametrization of the search space. We use _contrastive learning_ to identify networks across different initializations based on their data Jacobians, and automatically produce the first architecture embeddings independent from the parametrization of the search space. Using our contrastive embeddings, we show that traditional black-box optimization algorithms, without modification, can reach state-of-the-art performance in Neural Architecture Search. As our method provides a unified embedding space, we perform for the first time transfer learning between search spaces. Finally, we show the evolution of embeddings during training, motivating future investigations into using embeddings at different training stages to gain a deeper understanding of the networks in a search space.

## Requirements

- A `requirements.txt` file is available at the root of this repository, specifying the required packages for all of our experiments; 
- Python 3.7 is required as we make use of `importlib.resources`

## Reproducing our results

- To generate the Extended Projected Data Jacobian Matrices from a search space see `gen_epdjms.py`. These are necessary for all other scripts. 
-  To generate the transfer learning plots see `make_contrastive_transfer_plots.py`
-  To simulate NAS on a search space see `make_simulation.py`
-  To generate the t-SNE visualization of different stages of our method see `make_tsne_figs.py` 

## Citation 

If you found this code and findings useful in your research, please consider citing:
<Bibtex for citation>
  

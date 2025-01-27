# Investigating the Emergence of Complexity from the Dimensional Structure of Mental Representations

### Objective
This project assesses the ability of CLIP-Human Behavior Assessment (CLIP-HBA), which was created by fine-tuning CLIP-IQA on behavioral embeddings developed from human similarity judgements, to predict human visual complexity judgements.

### Prerequisites
Running this code requires:
- Python 3.8.18 (other versions may work, but this is the tested version)
```
git clone https://github.com/trishamazu/complexity-experiment-final.git
```
```
pip install -r requirements.txt
```
### Image Datasets
* [THINGS](https://things-initiative.org/) (Hebart et al., 2020)^[1]
* Bistable-Control
* [Savoias-Dataset](https://github.com/esaraee/Savoias-Dataset) (Saraee et al., 2018)^[2]. The Ground truth subfolder contains human complexity ratings.

### Embeddings
The embeddings folder contains two folders with embeddings for all three aforementioned datasets obtained from the respective models:
* [CLIP-CBA](https://github.com/datovar4/Image_Quality.git) (built using [CLIP-IQA](https://github.com/IceClear/CLIP-IQA.git) (Wang et al., 2022^[3])
* [CLIP-HBA](https://github.com/stephenczhao/CLIP-HBA-Finetune.git)

### Experiments
* The Experiments folder contains the code for the THINGS ranking experiment and the bistable 2-AFC experiment. 
* The Data subfolder contains the human complexity ratings for both experiments.

### Optimizations
* The Bayesian subfolder contains the code used to extract the optimal weights from a csv of embeddings and a target set of complexity scores. 
* The BestWeights subfolder contains the best weights obtained from running the optimization on the three datasets.
* HBAOptimizations and CBAOptimizations contain notebooks that can be used to run the optimization on different datasets. 
* The Neural subfolder contains code that can be used to create RDMs and compare them with THINGS MEG data.

[^1]: Zheng, C. Y., Pereira, F., Baker, C. I., & Hebart, M. N. (2019). Revealing interpretable object representations from human behavior. International Conference on Learning Representations (ICLR) 2019. arXiv. https://doi.org/10.48550/arXiv.1901.02915

[^2]: Copyright [2018] [Saraee et al.] Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

[^3]: Wang, J., Chan, K. C. K., & Loy, C. C. (2022). Exploring CLIP for assessing the look and feel of images. Proceedings of the AAAI Conference on Artificial Intelligence. arXiv. https://doi.org/10.48550/arXiv.2207.12396

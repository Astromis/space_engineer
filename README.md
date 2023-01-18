# Space Exploration Toolkit

This repo contaions various methods that allows you to profilate the embedding space. The tooklit is just a compilation of methods that was implemented in various papers.

Currently implemented metrics:
* IsoScore (https://github.com/bcbi-edu/p_eickhoff_isoscore)
  * Cosine score
  * Partition score (ALL-BUT-THE-TOP: SIMPLE AND EFFECTIVE POST-PROCESSING FOR WORD REPRESENTATIONS)
  * Varex score
  * Intrinsic dimensionality score
* Spatial Histogram (https://github.com/akalino/semantic-structural-sentences)
  * Hopkins test

Currently implemented post-processing:
* GEM (Parameter-free Sentence Embedding via Orthogonal Basis, https://github.com/ziyi-yang/GEM/)
* SIF (https://github.com/PrincetonML/SIF)
* All-but-the-top postprocessing (https://gist.github.com/lgalke/febaaa1313d9c11f3bc8240defed8390)

## IsoScore: Measuring the Uniformity of Vector Space Utilization
 *by William Rudman, Nate Gillman, Taylor Rayne, and Carsten Eickhoff*

IsoScore is a tool that measures how uniformly a point cloud utilizes the Euclidian space that it sits inside of. 
See the original paper ([https://arxiv.org/abs/2108.07344](https://arxiv.org/abs/2108.07344)) for more information. 
This repository contains the Python3 implementation of IsoScore.

### How to install

The only dependencies are `numpy` and `sklearn`.

```
pip install IsoScore
```

### How to use

If you want to compute the IsoScore for a point cloud <img src="https://render.githubusercontent.com/render/math?math=X">  that sits inside <img src="https://render.githubusercontent.com/render/math?math=\mathbb R^n">, then <img src="https://render.githubusercontent.com/render/math?math=X"> must be a `numpy` array of shape <img src="https://render.githubusercontent.com/render/math?math=(n,m)">, where <img src="https://render.githubusercontent.com/render/math?math=X"> contains <img src="https://render.githubusercontent.com/render/math?math=m"> points.
For example:


```python3
import numpy as np
from IsoScore import IsoScore

random_array_1 = np.random.normal(size=100)
random_array_2 = np.random.normal(size=100)
random_array_3 = np.random.normal(size=100)

# Computing the IsoScore for points sampled from a line (dim=1) in R^3
point_cloud_line = np.array([random_array_1, np.zeros(100), np.zeros(100)])
the_score = IsoScore.IsoScore(point_cloud_line)
print(f"IsoScore for 100 points sampled from this line in R^3 is {the_score}.")

# Computing the IsoScore for points sampled from a disk (dim=2) in R^3
point_cloud_disk = np.array([random_array_1, random_array_2, np.zeros(100)])
the_score = IsoScore.IsoScore(point_cloud_disk)
print(f"IsoScore for 100 points sampled from this disk in R^3 is {the_score}.")

# Computing the IsoScore for points sampled from a ball (dim=3) in R^3
point_cloud_ball = np.array([random_array_1, random_array_2, random_array_3])
the_score = IsoScore.IsoScore(point_cloud_ball)
print(f"IsoScore for 100 points sampled from this ball in R^3 is {the_score}.")
```

### Isotropy in Contextualized Embeddings
We obtain contextualized word embeddings for the WikiText-2 corpus using: https://github.com/TideDancer/IsotropyContxt.

The embedding_results directory contains isotropy scores for BERT, DistilBERT, GPT and GPT-2. 

### Visuals 
Please consult ```visuals.ipynb ``` to quickly run tests and recreate figures.  


### Citing

If you would like to cite this work, please refer to:
```bibtex
@Article{Rudman-Gillman-Rayne-Eickhoff-IsoScore,
    title = "IsoScore: Measuring the Uniformity of Vector Space Utilization",
    author =    {William Rudman and
                Nate Gillman and 
                Taylor Rayne and 
                Carsten Eickhoff},
    month = aug,
    year = "2021",
    url = "https://arxiv.org/abs/2108.07344",
}
```


### License

```
MIT License

Copyright (c) 2021 William Rudman, Nate Gillman, Taylor Rayne, Carsten Eickhoff

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Structure and Semantics of Sentence Embeddings 
This repository houses the experiments as detailed in the paper "A Comparative Study on Structural and Semantic Properties of Sentence Embeddings".
To run the experiments on your own machine, begin by creating the necessary datasets as detailed in the data/ directory. Next, generate the embedding sets for the knowledge graph in kg-embeddings/.
Generate all the sentence embeddings by following the README in the sentence-embeddings/ directory. 
To create the maps, follow the instructions in align/.
Finally, run the clustering experiments in cluster/.

### General components

Measure clusterability of embeddings and Hopkins statistics.

### References

```
@article{kalinowski-2020-structure-semantics-sentences,
         title="A Comparative Study on Structural and Semantic Properties of Sentence Embeddings",
         author="Kalinowski, Alexander and An, Yuan",
         journal="arXiv preprint",
         month="09",
         year="2020",
         url="update me"}
```
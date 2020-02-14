scikit-fusion
-------------

[![build: passing](https://img.shields.io/travis/marinkaz/scikit-fusion.svg)](https://travis-ci.org/marinkaz/scikit-fusion)
[![BSD license](https://img.shields.io/badge/License-BSD-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

*scikit-fusion* is a Python module for data fusion and learning over heterogeneous datasets. The core of scikit-fusion are recent collective latent factor models and large-scale joint matrix factorization algorithms. 

[**[News:]**](https://github.com/acopar/fast-fusion) [Fast CPU and GPU-accelerated implementatons](https://github.com/acopar/fast-fusion) of some of our methods.

[**[News:]**](https://github.com/marinkaz/scikit-fusion) [Scikit-fusion](https://github.com/marinkaz/scikit-fusion), collective latent factor models, matrix factorization for data fusion and learning over hetnets.

[**[News:]**](https://github.com/mims-harvard/fastGNMF) [fastGNMF](https://github.com/mims-harvard/fastGNMF), fast implementation of graph-regularized non-negative matrix factorization using [Facebook FAISS](https://github.com/facebookresearch/faiss).

<p align="center">
<img src="https://github.com/marinkaz/scikit-fusion/blob/master/fusion.png" width="800" align="center">
</p>

Dependencies
------------

scikit-fusion is tested to work under Python 3.

The required dependencies to build the software are Numpy >= 1.7, SciPy >= 0.12,
PyGraphviz >= 1.3 (needed only for drawing data fusion graphs) and Joblib >= 0.8.4.

Install
-------

This package uses distutils, which is the default way of installing
python modules. To install in your home directory, use:

    python setup.py install --user

To install for all users on Unix/Linux:

    python setup.py build
    sudo python setup.py install

For development mode use:

    python setup.py develop

Use
---

Let's generate three random data matrices describing three different object types:

     >>> import numpy as np
     >>> R12 = np.random.rand(50, 100)
     >>> R13 = np.random.rand(50, 40)
     >>> R23 = np.random.rand(100, 40)

Next, we define our data fusion graph:

     >>> from skfusion import fusion
     >>> t1 = fusion.ObjectType('Type 1', 10)
     >>> t2 = fusion.ObjectType('Type 2', 20)
     >>> t3 = fusion.ObjectType('Type 3', 30)
     >>> relations = [fusion.Relation(R12, t1, t2),
                      fusion.Relation(R13, t1, t3),
                      fusion.Relation(R23, t2, t3)]
     >>> fusion_graph = fusion.FusionGraph()
     >>> fusion_graph.add_relations_from(relations)

and then collectively infer the latent data model:

     >>> fuser = fusion.Dfmf()
     >>> fuser.fuse(fusion_graph)
     >>> print(fuser.factor(t1).shape)
     (50, 10)

Afterwards new data might arrive:

     >>> new_R12 = np.random.rand(10, 100)
     >>> new_R13 = np.random.rand(10, 40)

for which we define the fusion graph:

     >>> new_relations = [fusion.Relation(new_R12, t1, t2),
                          fusion.Relation(new_R13, t1, t3)]
     >>> new_graph = fusion.FusionGraph(new_relations)

and transform new objects to the latent space induced by the ``fuser``:

     >>> transformer = fusion.DfmfTransform()
     >>> transformer.transform(t1, new_graph, fuser)
     >>> print(transformer.factor(t1).shape)
     (10, 10)

****

scikit-fusion contains several applications of data fusion:

    >>> from skfusion import datasets
    >>> dicty = datasets.load_dicty()
    >>> print(dicty)
    FusionGraph(Object types: 3, Relations: 3)
    >>> print(dicty.object_types)
    {ObjectType(GO term), ObjectType(Experimental condition), ObjectType(Gene)}
    >>> print(dicty.relations)
    {Relation(ObjectType(Gene), ObjectType(GO term)),
     Relation(ObjectType(Gene), ObjectType(Gene)),
     Relation(ObjectType(Gene), ObjectType(Experimental condition))}

Selected publications (Methods)
------------------------------

- Data fusion by matrix factorization: http://dx.doi.org/10.1109/TPAMI.2014.2343973
- Jumping across biomedical contexts using compressive data fusion: https://academic.oup.com/bioinformatics/article/32/12/i90/2240593
- Survival regression by data fusion: http://www.tandfonline.com/doi/abs/10.1080/21628130.2015.1016702
- Gene network inference by fusing data from diverse distributions: https://academic.oup.com/bioinformatics/article/31/12/i230/216398
- Fast optimization of non-negative matrix tri-factorization: https://doi.org/10.1371/journal.pone.0217994

Selected publications (Applications)
------------------------------------

- A comprehensive structural, biochemical and biological profiling of the human NUDIX hydrolase family: https://www.nature.com/articles/s41467-017-01642-w
- Gene prioritization by compressive data fusion and chaining: http://dx.doi.org/10.1371/journal.pcbi.1004552
- Discovering disease-disease associations by fusing systems-level molecular data: http://www.nature.com/srep/2013/131115/srep03202/full/srep03202.html
- Matrix factorization-based data fusion for gene function prediction in baker's yeast and slime mold: http://www.worldscientific.com/doi/pdf/10.1142/9789814583220_0038
- Matrix factorization-based data fusion for drug-induced liver injury prediction: http://www.tandfonline.com/doi/abs/10.4161/sysb.29072
- Collective pairwise classification for multi-way analysis of disease and drug data: https://doi.org/10.1142/9789814749411_0008

Tutorials
---------

- Large-scale data fusion by collective matrix factorization, Basel Computational Biology Conference, [BC]^2 [[Slides]](http://helikoid.si/bc215/bc2-slides.pdf) [[Handouts]](http://helikoid.si/bc215/bc2-handouts.pdf)
- Data fusion of everything, 37th Annual International Conference of the IEEE Engineering in Medicine and Biology Society, EMBC [[Slides]](http://helikoid.si/embc15/embc-slides.pdf) [[Handouts]](http://helikoid.si/embc15/embc-handouts.pdf)

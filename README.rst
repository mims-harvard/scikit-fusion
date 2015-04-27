.. -*- mode: rst -*-

scikit-fusion
=============

|Travis|_

.. |Travis| image:: https://travis-ci.org/marinkaz/scikit-fusion.svg?branch=master
.. _Travis: https://travis-ci.org/marinkaz/scikit-fusion

scikit-fusion is a Python module for data fusion based on recent collective latent
factor models.

Dependencies
============

scikit-fusion is tested to work under Python 2.7+.

The required dependencies to build the software are Numpy >= 1.7, SciPy >= 0.12,
PyGraphviz >= 1.3 (needed only for drawing data fusion graphs) and Joblib >= 0.8.4.

Install
=======

This package uses distutils, which is the default way of installing
python modules. To install in your home directory, use::

    python setup.py install --user

To install for all users on Unix/Linux::

    python setup.py build
    sudo python setup.py install

For development mode use::

    python setup.py develop

Usage
=====

Let's generate three random data matrices describing three different object types::

     >>> import numpy as np
     >>> R12 = np.random.rand(50, 100)
     >>> R13 = np.random.rand(50, 40)
     >>> R23 = np.random.rand(100, 40)

Next, we define our data fusion graph::

     >>> from skfusion import fusion
     >>> t1 = fusion.ObjectType('type1', 10)
     >>> t2 = fusion.ObjectType('type2', 20)
     >>> t3 = fusion.ObjectType('type3', 30)
     >>> relations = [fusion.Relation(R12, t1, t2),
            fusion.Relation(R13, t1, t3), fusion.Relation(R23, t2, t3)]
     >>> fusion_graph = fusion.FusionGraph()
     >>> fusion_graph.add(relations)

and then collectively infer the latent data model::

     >>> fuser = fusion.Dfmf(fusion_graph)
     >>> fuser.fuse()
     >>> print(fuser.factor(t1).shape)
     (50, 10)


Afterwards new data might arrive::

     >>> new_R12 = np.random.rand(10, 100)
     >>> new_R13 = np.random.rand(10, 40)

for which we define the fusion graph::

     >>> new_relations = [fusion.Relation(new_R12, t1, t2),
            fusion.Relation(new_R13, t1, t3)]
     >>> new_graph = fusion.FusionGraph(new_relations)

and transform new objects to the latent space induced by the ``fuser``::

     >>> transformer = fusion.DfmfTransform(t1, new_graph, fuser)
     >>> transformer.transform()
     >>> print(transformer.factor(t1).shape)
     (10, 10)

****

scikit-fusion is distributed with few working data fusion scenarios::

    >>> from skfusion import datasets
    >>> dicty = datasets.load_dicty()
    >>> fuser = fusion.Dfmf()
    >>> fuser.add_relation(dicty.gene, dicty.go_term, dicty.ann.data)
    >>> fuser.add_relation(dicty.gene, dicty.exprc, dicty.expr.data)
    >>> fuser.add_constraint(dicty.gene, dicty.ppi.data)

Relevant links
==============

- Official source code repo: https://github.com/marinkaz/scikit-fusion
- HTML documentation: TBA
- Download releases: https://github.com/marinkaz/scikit-fusion/releases
- Issue tracker: https://github.com/marinkaz/scikit-fusion/issues

****

- Data fusion by matrix factorization: http://dx.doi.org/10.1109/TPAMI.2014.2343973
- Discovering disease-disease associations by fusing systems-level molecular data: http://www.nature.com/srep/2013/131115/srep03202/full/srep03202.html
- Matrix factorization-based data fusion for gene function prediction in baker's yeast and slime mold: http://www.worldscientific.com/doi/pdf/10.1142/9789814583220_0038
- Matrix factorization-based data fusion for drug-induced liver injury prediction: http://www.tandfonline.com/doi/abs/10.4161/sysb.29072
- Survival regression by data fusion: http://www.tandfonline.com/doi/abs/10.1080/21628130.2015.1016702

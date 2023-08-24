# CS224u: Natural Language Understanding

Code for [the Stanford course](http://web.stanford.edu/class/cs224u/).

Spring 2023

[Christopher Potts](http://web.stanford.edu/~cgpotts/)


## Core components


### `setup.ipynb`

Details on how to get set up to work with this code.


### `hw_*.ipynb`

The set of homeworks for the current run of the course.


### `tutorial_*` notebooks

Introductions to Jupyter notebooks, scientific computing with NumPy and friends, and PyTorch.


### `torch_*.py` modules

A generic optimization class (`torch_model_base.py`) and subclasses for GloVe, Autoencoders, shallow neural classifiers, RNN classifiers, tree-structured networks, and grounded natural language generation.

`tutorial_pytorch_models.ipynb` shows how to use these modules as a general framework for creating original systems.


### `evaluation_*.ipynb` and `projects.md`

Notebooks covering key experimental methods and practical considerations, and tips on writing up and presenting work in the field.


### `iit*` and `feature_attribution.ipynb`

Part of our unit on explainability and model analysis.


### `np_*.py` modules

This is now considered background material for the course.

Reference implementations for the `torch_*.py` models, designed to reveal more about how the optimization process works.


### `vsm_*`

This is now considered background material for the course.

A unit on vector space models of meaning, covering traditional methods like PMI and LSA as well as newer methods like Autoencoders and GloVe. `vsm.py` provides a lot of the core functionality, and `torch_glove.py` and `torch_autoencoder.py` are the learned models that we cover. `vsm_03_contextualreps.ipynb` explores methods for deriving static representations from contextual models.


### `sst_*`

This is now considered background material for the course.

A unit on sentiment analysis with the [English Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/treebank.html). The core code is `sst.py`, which includes a flexible experimental framework. All the PyTorch classifiers are put to use as well: `torch_shallow_neural_network.py`, `torch_rnn_classifier.py`, and `torch_tree_nn.py`.


### `finetuning.ipynb`

This is now considered background material for the course.

Using pretrained parameters from [Hugging Face](https://huggingface.co) for featurization and fine-tuning.


### `utils.py`

Miscellaneous core functions used throughout the code.


### `test/`

To run these tests, use

```py.test -vv test/*```

or, for just the tests in `test_shallow_neural_classifiers.py`,

```py.test -vv test/test_shallow_neural_classifiers.py```

If the above commands don't work, try

```python3 -m pytest -vv test/test_shallow_neural_classifiers.py```


## License

The materials in this repo are licensed under the [Apache 2.0 license](LICENSE) and a [Creative Commons Attribution-ShareAlike 4.0 International license](http://creativecommons.org/licenses/by-sa/4.0/).

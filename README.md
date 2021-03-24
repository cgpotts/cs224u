# CS224u: Natural Language Understanding

Code for [the Stanford course](http://web.stanford.edu/class/cs224u/).

Spring 2021

# Instructors

* [Bill MacCartney](http://nlp.stanford.edu/~wcmac/)
* [Christopher Potts](http://web.stanford.edu/~cgpotts/)


# Core components


## `setup.ipynb`

Details on how to get set up to work with this code.


## `tutorial_*` notebooks

Introductions to Juypter notebooks, scientific computing with NumPy and friends, and PyTorch.


## `torch_*.py` modules

A generic optimization class (`torch_model_base.py`) and subclasses for GloVe, Autoencoders, shallow neural classifiers, RNN classifiers, tree-structured networks, and grounded natural language generation.

`tutorial_pytorch_models.ipynb` shows how to use these modules as a general framework for creating original systems.


## `np_*.py` modules

Reference implementations for the `torch_*.py` models, designed to reveal more about how the optimization process works.


## `vsm_*` and `hw_wordrelatedness.ipynb`

A until on vector space models of meaning, covering traditional methods like PMI and LSA as well as newer methods like Autoencoders and GloVe. `vsm.py` provides a lot of the core functionality, and `torch_glove.py` and `torch_autoencoder.py` are the learned models that we cover. `vsm_03_retroffiting.ipynb` is an extension that uses `retrofitting.py`, and `vsm_04_contextualreps.ipynb` explores methods for deriving static representations from contextual models.


## `sst_*` and `hw_sst.ipynb`

A unit on sentiment analysis with the [English Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/treebank.html). The core code is `sst.py`, which includes a flexible experimental framework. All the PyTorch classifiers are put to use as well: `torch_shallow_neural_network.py`, `torch_rnn_classifier.py`, and `torch_tree_nn.py`.


## `rel_ext*` and `hw_rel_ext.ipynb`

A unit on relation extraction with distant supervision.


## `nli_*` and `hw_wordentail.ipynb`

A unit on Natural Language Inference. `nli.py` provides core interfaces to a variety of NLI dataset, and an experimental framework. All the PyTorch classifiers are again in heavy use: `torch_shallow_neural_network.py`, `torch_rnn_classifier.py`, and `torch_tree_nn.py`.


## `colors*`, `torch_color_describer.py`, and `hw_colors.ipynb`

A unit on grounded natural language generation, focused on generating context-dependent color descriptions using the [English Stanford Colors in Context dataset](https://cocolab.stanford.edu/datasets/colors.html).


## `finetuning.ipynb`

Using pretrained parameters from [Hugging Face](https://huggingface.co) for featurization and fine-tuning.


## `evaluation_*.ipynb` and `projects.md`

Notebooks covering key experimental methods and practical considerations, and tips on writing up and presenting work in the field.


## `utils.py`

Miscellaneous core functions used throughout the code.


## `test/`

To run these tests, use

```py.test -vv test/*```

or, for just the tests in `test_shallow_neural_classifiers.py`,

```py.test -vv test/test_shallow_neural_classifiers.py```

If the above commands don't work, try

```python3 -m pytest -vv test/test_shallow_neural_classifiers.py```


## License

The materials in this repo are licensed under the [Apache 2.0 license](LICENSE) and a [Creative Commons Attribution-ShareAlike 4.0 International license](http://creativecommons.org/licenses/by-sa/4.0/).

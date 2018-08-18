# NLP: Experimenting-Word2Vec-algorithm-on-Finnish-Text

Use the Word2Vec skip-gram model to embed textual Finnish words to numerical vectors which is the basis for almost all kind of NLP (Natural Language Processing) tasks. Implement the Word2Vec skip-gram model with tensorflow. Tokenize and lemmatize Finnish texts with the [voikko](https://voikko.puimula.org/) library which seems to be the only NLP linguistic tool that can morphologically analyze Finnish texts.

### Libraries used

1. tensorflow

2. voikko

3. numpy

### Dataset
The dataset used in this experiment is extracted from News and is intended to be used in a machine translation task. It is used to train the word2vec model herein. Can be downloaded from [here](http://statmt.org/wmt17/translation-task.html#download).


### Descriptions of files

#### Folder: notebooks

fin_w2v_utils.ipynb: Functions to preprocess textual data (e.g. tokenization, lemmatization) in order to construct training inputs for the word2vec skip-gram model.

fin_word2vec.ipynb: Implementation of word2vec skip-gram model with NCE loss.

voikko_install.ipynb: Instuctions to install voikko library **libvoikko** on mac.

voikko_test.ipynb: Test the usage of voikko to tokenize a sentence and lemmatize a document.

#### Folder: python

fin_w2v_utils.py: python version of fin_w2v_utils.ipynb.

fin_word2vec.py: python version of fin_word2vec.ipynb.

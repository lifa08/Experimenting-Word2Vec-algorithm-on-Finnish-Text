# ![Word2VecFin](notebooks/w2v_model.png)
# Word2Vec on Finnish Text

Experiment Word2Vec skip-gram model to embed textual Finnish words to numerical vectors which is the basis for almost all kind of NLP (Natural Language Processing) tasks. Implement the Word2Vec skip-gram model with tensorflow. Tokenize and lemmatize Finnish texts with the [voikko](https://voikko.puimula.org/) library which seems to be the only NLP linguistic tool that can morphologically analyze Finnish texts.

### Libraries used

1. tensorflow

2. voikko

3. numpy

### Dataset
* Extracted from News and is intended to be used in a machine translation task
* Download [link](http://statmt.org/wmt17/translation-task.html#download).


### Descriptions of files

#### [notebooks](notebooks/)

[fin_w2v_utils.ipynb](notebooks/fin_w2v_utils.ipynb)

Preprocess textual data (e.g. tokenization, lemmatization) in order to construct training inputs for the word2vec skip-gram model.

[fin_word2vec.ipynb](notebooks/fin_word2vec.ipynb)

Implement the word2vec skip-gram model with NCE loss.

[voikko_install.ipynb](notebooks/voikko_install.ipynb)

Include instuctions to install voikko library **libvoikko** on mac.

[voikko_test.ipynb](notebooks/voikko_test.ipynb)

Test the usage of voikko to tokenize a sentence and lemmatize a document.

#### [python](python/)

[fin_w2v_utils.py](python/fin_w2v_utils.py): python version of [fin_w2v_utils.ipynb](notebooks/fin_w2v_utils.ipynb)

[fin_word2vec.py](python/fin_word2vec.py): python version of [fin_word2vec.ipynb](notebooks/fin_word2vec.ipynb)

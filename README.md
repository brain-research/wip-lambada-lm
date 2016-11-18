An LSTM language model on LAMBADA dataset
=========================================

DISCLAIMER: this is not an official Google product.

This repository contain an implementation of LSTM language model.
It is tested on the LAMBADA dataset, the results are consistent
with the original paper [1]. The language model guesses are
often correct for CONTROL dataset and almost always wrong
for VALID and TEST datasets.

The file to run is ``lambada_lm/lm.py``

[1] https://arxiv.org/pdf/1606.06031v1.pdf

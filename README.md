# seq2seq-Machine-Translator

# What is Machine Translator:
Machine Translator is a task to translate a sentence from one language to another language. This research began in the early 1950s and systems were mostly rule-based. During the 1990s-2010s Statistical Machine Translation (SMT) was popular. The core idea of SMT is that we need to learn the probabilistic model from data. We need a large amount of parallel data for the translation model. SMT system can be divided into two main components: Translation Model and Language Model. In SMT, alignment is a major issue. Alignment is the correspondence between particular words in the translated sentence pair. Alignment is an issue because some words have no counterpart and it can be many to one, one to many, and even many to many. To search for the best translation, she suggested using a heuristic search algorithm. This process is called decoding. But SMT requires extra resources and human effort to compile and maintain

Neural Machine Translator (NMT) is a way to do Machine Translation with a single neural network.This architecture is called sequence-to-sequence (or seq2seq) and it involves two RNNs. Here an Encoder is used to produce encoding of the source sentence. Then the encoding is passed to the decoder of the RNN. The decoder is an LM that generates a target sentence, conditioned on encoding. So, seq2seq is a conditional LM because it predicts the next word of a target sentence and is conditioned on the source sentence. This happens during the testing period. During training, we need to feed the source sentence into the encoder RNN and then feed the target sentence into the decoder RNN and pass the final hidden state of the encoder to be the first hidden state of the decoder

# Bilingual Evaluation Understudy (BLEU):
Bilingual Evaluation Understudy (BLEU) is used to evaluate NMT. It compares machine-written translation to one or several human-written translations and computes a similarity score. It is useful but imperfect as there are many ways to translate a sentence.MT is not solved yet as it picks up biases in training data, and it has less interpretability. So, research is going on MT.


# DataSet:
English-Spanish dataset is used from this data source: https://www.statmt.org/europarl/


# Inspired by:

CS 5642  Advanced NLP and https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html 

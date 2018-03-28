# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Batch reader to seq2seq attention model, with bucketing support."""

from collections import namedtuple
import Queue
from threading import Thread

import numpy as np
import tensorflow as tf

import data

ModelInput = namedtuple('ModelInput',
                        'enc_input enc_len topic_vector origin_article origin_abstract')

QUEUE_NUM_BATCH = 1000


class Batcher(object):
    """Batch reader with shuffling and bucketing support."""

    def __init__(self, data_path, vocab, hps,
                 article_key, abstract_key, topic_key, max_article_sentences,
                 max_abstract_sentences, bucketing=True, truncate_input=False):
        """Batcher constructor.

        Args:
          data_path: tf.Example filepattern.
          vocab: Vocabulary.
          hps: Seq2SeqAttention model hyperparameters.
          article_key: article feature key in tf.Example.
          abstract_key: abstract feature key in tf.Example.
          max_article_sentences: Max number of sentences used from article.
          max_abstract_sentences: Max number of sentences used from abstract.
          bucketing: Whether bucket articles of similar length into the same batch.
          truncate_input: Whether to truncate input that is too long. Alternative is
            to discard such examples.
        """
        self._data_path = data_path
        self._vocab = vocab
        self._hps = hps
        self._article_key = article_key
        self._abstract_key = abstract_key
        self._topic_key = topic_key
        self._max_article_sentences = max_article_sentences
        self._max_abstract_sentences = max_abstract_sentences
        self._bucketing = bucketing
        self._truncate_input = truncate_input
        self._input_queue = Queue.Queue(QUEUE_NUM_BATCH)
        self._input_threads = []
        self._input_threads.append(Thread(target=self._FillInputQueue))
        self._input_threads[-1].daemon = True
        self._input_threads[-1].start()

    def NextBatch(self):
        """Returns a batch of inputs for seq2seq attention model.

        Returns:
          enc_batch: A batch of encoder inputs [batch_size, hps.enc_timestamps].
          dec_batch: A batch of decoder inputs [batch_size, hps.dec_timestamps].
          target_batch: A batch of targets [batch_size, hps.dec_timestamps].
          enc_input_len: encoder input lengths of the batch.
          dec_input_len: decoder input lengths of the batch.
          loss_weights: weights for loss function, 1 if not padded, 0 if padded.
          origin_articles: original article words.
          origin_abstracts: original abstract words.
        """
        enc_batch = np.zeros(
            (self._hps.batch_size, self._hps.enc_timesteps), dtype=np.int32)
        enc_input_lens = np.zeros(self._hps.batch_size, dtype=np.int32)
        # enc_keyword_ids = np.zeros(self._hps.batch_size, dtype=np.int32)
        topic_batch = np.zeros((self._hps.batch_size, self._hps.num_hidden),
                               dtype=np.float32)

        inputs = self._input_queue.get()
        (enc_inputs, enc_input_len, topic_vector, article, abstract) = inputs

        enc_input_lens[:] = enc_input_len
        # enc_keyword_ids[:] = enc_keyword_ids
        enc_batch[:] = enc_inputs[:]
        topic_batch[:] = topic_vector[:]

        return enc_batch, topic_batch, enc_input_lens, article, abstract

    def _FillInputQueue(self):
        """Fill input queue with ModelInput."""
        # start_id = self._vocab.WordToId(data.SENTENCE_START)
        pad_id = self._vocab.WordToId(data.PAD_TOKEN)
        input_gen = self._TextGenerator(data.ExampleGen(self._data_path, 1))
        unknow_id = self._vocab.WordToId(data.UNKNOWN_TOKEN)
        while True:
            try:
                (article, abstract, topic) = input_gen.next()
            except (GeneratorExit, StopIteration):
                break
            article_sentences = [sent.strip() for sent in
                                 data.ToSentences(article, include_token=False)]
            abstract_sentences = [sent.strip() for sent in
                                  data.ToSentences(abstract, include_token=False)]

            topic_list = ((topic.strip()).split())
            topic_vector = np.array(map(float, topic_list))
            if len(topic_vector) != self._hps.num_hidden:
                continue

            enc_inputs = []

            # Convert first N sentences to word IDs, stripping existing <s> and </s>.
            for i in xrange(min(self._max_article_sentences,
                                len(article_sentences))):
                enc_inputs += data.GetWordIds(article_sentences[i], self._vocab)

            # Filter out too-short input
            if len(enc_inputs) < self._hps.min_input_len:
                # tf.logging.warning('Drop an example - too short.\nenc:%d',
                #                   len(enc_inputs))
                continue

            # If we're not truncating input, throw out too-long input
            if not self._truncate_input:
                if len(enc_inputs) > self._hps.enc_timesteps:
                    # tf.logging.warning('Drop an example - too long.\nenc:%d',
                    #                   len(enc_inputs))
                    continue
            # If we are truncating input, do so if necessary
            else:
                if len(enc_inputs) > self._hps.enc_timesteps:
                    enc_inputs = enc_inputs[:self._hps.enc_timesteps]

            enc_input_len = len(enc_inputs)

            # Pad if necessary
            while len(enc_inputs) < self._hps.enc_timesteps:
                enc_inputs.append(pad_id)

            element = ModelInput(enc_inputs, enc_input_len, topic_vector,
                                 ' '.join(article_sentences),
                                 ' '.join(abstract_sentences))
            self._input_queue.put(element)

    def _TextGenerator(self, example_gen):
        """Generates article and abstract text from tf.Example."""
        while True:
            try:
                e = example_gen.next()
            except (GeneratorExit, StopIteration):
                break
            try:
                article_text = self._GetExFeatureText(e, self._article_key)
                abstract_text = self._GetExFeatureText(e, self._abstract_key)
                topic_text = self._GetExFeatureText(e, self._topic_key)
            except ValueError:
                tf.logging.error('Failed to get article or abstract from example')
                continue

            yield (article_text, abstract_text, topic_text)

    def _GetExFeatureText(self, ex, key):
        """Extract text for a feature from td.Example.

        Args:
          ex: tf.Example.
          key: key of the feature to be extracted.
        Returns:
          feature: a feature text extracted.
        """
        return ex.features.feature[key].bytes_list.value[0]

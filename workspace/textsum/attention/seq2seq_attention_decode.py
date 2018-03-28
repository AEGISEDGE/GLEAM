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

"""Module for decoding."""

import os
import time

import tensorflow as tf
import beam_search
import data

FLAGS = tf.app.flags.FLAGS

DECODE_IO_FLUSH_INTERVAL = 60


class DecodeIO(object):
    """Writes the decoded and references to RKV files for Rouge score.

      See nlp/common/utils/internal/rkv_parser.py for detail about rkv file.
    """

    def __init__(self, outdir):
        self._cnt = 0
        self._outdir = outdir
        if not os.path.exists(self._outdir):
            os.mkdir(self._outdir)
        self._ref_file = None
        self._decode_file = None
        self._attention_file = None
        self._article_file = None

    def Write(self, reference, decode, attention, article):
        """Writes the reference and decoded outputs to RKV files.

        Args:
          reference: The human (correct) result.
          decode: The machine-generated result
          attention:
          article:
        """
        self._ref_file.write('output=%s\n' % reference)
        self._decode_file.write('output=%s\n' % decode)
        self._attention_file.write('%s\n%s\n' % (decode.split(), attention))
        self._article_file.write('%s\n' % article.split())
        self._cnt += 1
        if self._cnt % DECODE_IO_FLUSH_INTERVAL == 0:
            self._ref_file.flush()
            self._decode_file.flush()
            self._attention_file.flush()
            self._article_file.flush()

    def ResetFiles(self):
        """Resets the output files. Must be called once before Write()."""
        if self._ref_file: self._ref_file.close()
        if self._decode_file: self._decode_file.close()
        timestamp = int(time.time())
        self._ref_file = open(
            os.path.join(self._outdir, 'ref%d' % timestamp), 'w')
        self._decode_file = open(
            os.path.join(self._outdir, 'decode%d' % timestamp), 'w')
        self._attention_file = open(
            os.path.join(self._outdir, 'attention%d' % timestamp), 'w')
        self._article_file = open(
            os.path.join(self._outdir, 'article%d' % timestamp), 'w')


class BSDecoder(object):
    """Beam search decoder."""

    def __init__(self, model, dec_reader, hps, vocab):
        """Beam search decoding.

        Args:
          model: The seq2seq attentional model.
          batch_reader: The batch data reader.
          hps: Hyperparamters.
          vocab: Vocabulary
        """
        self._model = model
        self._model.build_graph()
        self._dec_reader = dec_reader
        self._hps = hps
        self._vocab = vocab
        self._saver = tf.train.Saver()
        self._decode_io = DecodeIO(FLAGS.decode_dir)

    def DecodeLoop(self):
        """Decoding loop for long running process."""
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True  # don't use too much resource
        gpu_config.allow_soft_placement = True  # if choose one don't exit,automatically choose one
        gpu_config.log_device_placement = True  # use your choose GPU
        sess = tf.Session(config=gpu_config)
        self._Decode(self._saver, sess)

    def _Decode(self, saver, sess):
        """Restore a checkpoint and decode it.

        Args:
          saver: Tensorflow checkpoint saver.
          sess: Tensorflow session.
        Returns:
          If success, returns true, otherwise, false.
        """
        ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
        if not (ckpt_state and ckpt_state.model_checkpoint_path):
            tf.logging.info('No model to decode yet at %s', FLAGS.log_root)
            return False

        tf.logging.info('checkpoint path %s', ckpt_state.model_checkpoint_path)
        ckpt_path = os.path.join(
            FLAGS.log_root, os.path.basename(ckpt_state.model_checkpoint_path))
        tf.logging.info('renamed checkpoint path %s', ckpt_path)
        saver.restore(sess, ckpt_path)

        self._decode_io.ResetFiles()
        while True:
            if self._dec_reader._input_queue.empty():
                break
            (article, article_lens, origin_articles,
             origin_abstracts) = self._dec_reader.NextBatch()
            bs = beam_search.BeamSearch(
                self._model, self._hps.batch_size,
                self._vocab.WordToId(data.SENTENCE_START),
                self._vocab.WordToId(data.SENTENCE_END),
                self._hps.dec_timesteps)

            best_beam = bs.BeamSearch(sess, article, article_lens)[0]
            decode_output = [int(t) for t in best_beam.tokens[1:]]
            attention_list = best_beam.attentions
            self._DecodeBatch(
                origin_articles, origin_abstracts, decode_output, attention_list)

    def _DecodeBatch(self, article, abstract, output_ids, attentions):
        """Convert id to words and writing results.

        Args:
          article: The original article string.
          abstract: The human (correct) abstract string.
          output_ids: The abstract word ids output by machine.
        """

        decoded_output = ' '.join(data.Ids2Words(output_ids, self._vocab))
        end_p = decoded_output.find(data.SENTENCE_END, 0)
        if end_p != -1:
            decoded_output = decoded_output[:end_p]
        tf.logging.info('article:  %s', article)
        tf.logging.info('abstract: %s', abstract)
        tf.logging.info('decoded:  %s', decoded_output)
        self._decode_io.Write(abstract, decoded_output.strip(), attentions, article)

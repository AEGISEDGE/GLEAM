"""generate vocabulary.
Usage:
python batch_text_vector.py souce_dir target_dir vocab_path
"""

import os
import re
import sys
from collections import Counter

reg_p = r'article=<d> <p> <s>.*?</s> </p> </d>'

# SENTENCE_START = '<s>'
# SENTENCE_END = '</s>'
UNKNOWN_TOKEN = '<UNK>'
PAD_TOKEN = '<PAD>'
START_VOCAB = [UNKNOWN_TOKEN, PAD_TOKEN]


class Vocab(object):
    """Vocabulary class for mapping words and ids."""

    def __init__(self, vocab_file, max_size):
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0

        with open(vocab_file, 'r') as vocab_f:
            for line in vocab_f:
                pieces = line.split()
                if len(pieces) != 2:
                    sys.stderr.write('Bad line: %s\n' % line)
                    continue
                if pieces[0] in self._word_to_id:
                    raise ValueError('Duplicated word: %s.' % pieces[0])
                self._word_to_id[pieces[0]] = self._count
                self._id_to_word[self._count] = pieces[0]
                self._count += 1
                if self._count >= max_size:
                    break
                    # raise ValueError('Too many words: >%d.' % max_size)

    def WordToId(self, word):
        if word not in self._word_to_id:
            return self._word_to_id[UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def IdToWord(self, word_id):
        if word_id not in self._id_to_word:
            raise ValueError('id not found in vocab: %d.' % word_id)
        return self._id_to_word[word_id]

    def NumIds(self):
        return self._count


def generate_vector(list_sentence, vocab):
    # give a string generate the vector
    data = []
    word_count = []
    for sentence in list_sentence:
        doc = {}
        count = 0
        sentence = str(sentence)[20:-14]
        sentence_word = sentence.split()
        word_count_dict = dict(Counter(sentence_word))
        for key in word_count_dict:
            if key in vocab._word_to_id:
                doc[vocab.WordToId(key)] = word_count_dict[key]
                count += word_count_dict[key]
        data.append(doc)
        word_count.append(count)
    return data, word_count


def generate_vocab():
    # generate traing data
    if not os.path.exists(sys.argv[2]):
        os.makedirs(sys.argv[2])

    vocab = Vocab(sys.argv[3],10000)
    vocab_path = os.path.join(sys.argv[2], 'train.feat')
    wf = open(vocab_path, 'w')

    pattern_p = re.compile(reg_p)
    for root, path, files in os.walk(sys.argv[1]):
        for f in files:
            rf = open(os.path.join(root, f), 'r')
            file_context = rf.read()
            rf.close()
            all_sentence = pattern_p.findall(file_context)
            all_str = ""
            for sentence in all_sentence:
                new_sentence = "1"
                sentence = str(sentence)[20:-14]

                sentence_word = sentence.split()
                word_count_dict = dict(Counter(sentence_word))
                for key in word_count_dict:
                    if key in vocab._word_to_id:
                        new_sentence += " " + str(vocab.WordToId(key)+1) + ":" \
                                        + str(word_count_dict[key])
                new_sentence += "\n"
                all_str += new_sentence
            wf.write(all_str)
    wf.close()


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print 'python batch_text_vector.py souce_dir target_dir vocab_path'
        sys.exit(-1)
    generate_vocab()

"""generate vocabulary.
Usage:
python vocab_headline.py souce_dir target_dir
"""

import os
import re
import sys

reg_p = r'abstract=<d> <p> <s>.*?</s>'

# SENTENCE_START = '<s>'
# SENTENCE_END = '</s>'
UNKNOWN_TOKEN = '<UNK>'
PAD_TOKEN = '<PAD>'
START_VOCAB = [UNKNOWN_TOKEN, PAD_TOKEN]


def generate_vocab():
    if not os.path.exists(sys.argv[2]):
        os.makedirs(sys.argv[2])

    pattern_p = re.compile(reg_p)
    vocab = {}
    for root, path, files in os.walk(sys.argv[1]):
        for f in files:
            rf = open(os.path.join(root, f), 'r')
            file_context = rf.read()
            rf.close()
            all_sentence = pattern_p.findall(file_context)
            all_str = ""
            for sentence in all_sentence:
                sentence = str(sentence)[21:-4]
                all_str += sentence
            words = all_str.split()
            for word in words:
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1
    vocab_list = sorted(vocab, key=vocab.get, reverse=True)
    vocab_path = os.path.join(sys.argv[2], 'vocab')
    all_vocab = ""
    # for w in START_VOCAB:
    #     all_vocab += w + ' 1\n'
    for w in vocab_list:
        all_vocab += w + ' ' + str(vocab[w]) + '\n'
    wf = open(vocab_path, 'w')
    wf.write(all_vocab)
    wf.close()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print 'python vocab_headline.py souce_dir target_dir'
        sys.exit(-1)
    generate_vocab()

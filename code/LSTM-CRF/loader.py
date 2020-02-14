import os
import re
import random
import codecs
from utils import create_dico, create_mapping
from utils import iob2, iob_iobes
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras_contrib.layers import CRF


def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub('\d', '0', s)


def load_sentences(path, zeros):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.

    zeros - Replace digits with 0

    """
    sentences = []
    sentence = []
    counter = 0
    for line in codecs.open(path, 'r', 'utf8'):
        counter += 1
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            word = line.split()
            #if len(word) < 2:
            #    print(word)
            if len(word) >= 2:
                sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences


def update_tag_scheme(sentences, tag_scheme):
    """
    Check and update sentences tagging scheme to IOB2.
    Only IOB1 and IOB2 schemes are accepted.
    """
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]
        # Check that tags are given in the IOB format
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            # raise Exception('Sentences should be given in IOB format! ' + 'Please check sentence %i:\n%s' % (i, s_str))
            print('Removing Problematic sentence: %i:\n%s' % (i, s_str))
            continue
        if tag_scheme == 'iob':
            # If format was IOB1, we convert to IOB2
            for word, new_tag in zip(s, tags):
                word[-1] = new_tag
        elif tag_scheme == 'iobes':
            new_tags = iob_iobes(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        else:
            raise Exception('Unknown tagging scheme!')


def word_mapping(sentences, lower):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    words = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
    dico = create_dico(words)
    dico['[MASK]'] = 30000000
    dico['[UNK]'] = 20000000
    dico['[PAD]'] = 10000000

    word_to_id, id_to_word = create_mapping(dico)
    print ("Found %i unique words (%i in total)" % (len(dico), sum(len(x) for x in words)))
    return dico, word_to_id, id_to_word


def char_mapping(sentences):
    """
    Create a dictionary and mapping of characters, sorted by frequency.
    """
    chars = ["".join([w[0] for w in s]) for s in sentences]
    dico = create_dico(chars)
    dico["[PAD]"] = 10000000
    #dico['[UNK]'] = 20000000
    char_to_id, id_to_char = create_mapping(dico)
    print ("Found %i unique characters" % len(dico))
    return dico, char_to_id, id_to_char


def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [[word[-1] for word in s] for s in sentences]
    dico = create_dico(tags)
    dico["[PAD]"] = 10000000
    #dico['[UNK]'] = 20000000
    tag_to_id, id_to_tag = create_mapping(dico)
    print ("Found %i unique named entity tags" % len(dico))
    return dico, tag_to_id, id_to_tag

def augment_with_pretrained_new(dictionary, model, words):
    """
    Augment the dictionary with words that have a pretrained embedding.
    If `words` is None, we add every word that has a pretrained embedding
    to the dictionary, otherwise, we only add the words that are given by
    `words` (typically the words in the development and test sets.)
    """
    #print('Loading pretrained embeddings from %s...' % ext_emb_path)
    #assert os.path.isfile(ext_emb_path)

    # Load pretrained embeddings from file
    pretrained = set(list(model.wv.vocab))
    # We either add every word in the pretrained file,
    # or only words given in the `words` list to which
    # we can assign a pretrained embedding
    if words is None:
        for word in pretrained:
            if word not in dictionary:
                dictionary[word] = 0
    else:
        for word in words:
            if any(x in pretrained for x in [
                word,
                word.lower(),
                re.sub('\d', '0', word.lower())
            ]) and word not in dictionary:
                dictionary[word] = 0

    word_to_id, id_to_word = create_mapping(dictionary)
    return dictionary, word_to_id, id_to_word


def augment_with_pretrained(dictionary, ext_emb_path, words):
    """
    Augment the dictionary with words that have a pretrained embedding.
    If `words` is None, we add every word that has a pretrained embedding
    to the dictionary, otherwise, we only add the words that are given by
    `words` (typically the words in the development and test sets.)
    """
    print('Loading pretrained embeddings from %s...' % ext_emb_path)
    assert os.path.isfile(ext_emb_path)

    # Load pretrained embeddings from file
    pretrained = set([
        line.rstrip().split()[0].strip()
        for line in codecs.open(ext_emb_path, 'r', 'utf-8')
        if len(ext_emb_path) > 0
    ])

    # We either add every word in the pretrained file,
    # or only words given in the `words` list to which
    # we can assign a pretrained embedding
    if words is None:
        for word in pretrained:
            if word not in dictionary:
                dictionary[word] = 0
    else:
        for word in words:
            if any(x in pretrained for x in [
                word,
                word.lower(),
                re.sub('\d', '0', word.lower())
            ]) and word not in dictionary:
                dictionary[word] = 0

    word_to_id, id_to_word = create_mapping(dictionary)
    return dictionary, word_to_id, id_to_word


def cap_feature(s):
    """
    Capitalization feature:
    1 = low caps
    2 = all caps
    3 = first letter caps
    4 = one capital (not first letter)
    """
    if s.lower() == s:
        return 1
    elif s.upper() == s:
        return 2
    elif s[0].upper() == s[0]:
        return 3
    else:
        return 4


def prepare_dataset(sentences, word_dim, char_dim, word_to_id, id_to_word, char_to_id, tag_to_id, lower=False, mask= True, mask_per = 0.2, train = False):
    def f(x):
        return x.lower() if lower else x

    def fs(x):
        if mask:
            if x[1] != 'O':
                if random.randint(0, 100)<(100*mask_per):
                    return '[UNK]'
        return x[0].lower() if lower else x[0]
    if train:
        words = [[word_to_id[fs(w) if f(w[0]) in word_to_id else '[UNK]'] for w in s] for s in sentences]
    else:
        words = [[word_to_id[f(w[0]) if f(w[0]) in word_to_id else '[UNK]'] for w in s] for s in sentences]
    words = pad_sequences(maxlen=word_dim, sequences=words, value=word_to_id["[PAD]"], padding='post', truncating='post')
    temp_words = [[id_to_word[w] for w in s] for s in words]

    chars = []
    for sentence in sentences:
        sent_seq = []
        for i in range(word_dim):
            word_seq = []
            for j in range(char_dim):
                try:
                    word_seq.append(char_to_id.get(sentence[i][0][j]))
                except:
                    word_seq.append(char_to_id.get("[PAD]"))
            sent_seq.append(word_seq)
        chars.append(np.array(sent_seq))
    caps = [[cap_feature(w[0]) for w in s] for s in sentences]
    caps = pad_sequences(maxlen=word_dim, sequences=caps, value=0, padding='post', truncating='post')
    tags = [[tag_to_id[w[-1]] for w in s] for s in sentences]
    tags = pad_sequences(maxlen=word_dim, sequences=tags, value=tag_to_id["[PAD]"], padding='post', truncating='post')
    return words, chars, caps, tags


def create_custom_objects():

    instance_holder = {"instance": None}

    class ClassWrapper(CRF):
        def __init__(self, *args, **kwargs):
            instance_holder["instance"] = self
            super(ClassWrapper, self).__init__(*args, **kwargs)

    def loss(*args):
        method = getattr(instance_holder["instance"], "loss_function")
        return method(*args)

    def accuracy(*args):
        method = getattr(instance_holder["instance"], "viterbi_acc")
        return method(*args)

    return {"CRF": ClassWrapper, "crf_loss": loss, "crf_viterbi_accuracy":accuracy}

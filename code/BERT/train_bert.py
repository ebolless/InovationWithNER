import os
import time
from collections import OrderedDict
import itertools
import pandas as pd
from our_model import bulid
from our_model_bert import bulid as build_bert
from keras.utils import plot_model, to_categorical
from keras.models import load_model
from utils import models_path, evaluate, eval_script, eval_temp, evaluate_auc
from loader import load_sentences, update_tag_scheme, word_mapping, char_mapping, tag_mapping, augment_with_pretrained,\
    prepare_dataset, create_custom_objects, augment_with_pretrained_new
import pickle
from gensim.models import Word2Vec
from flair.data import Sentence
import numpy as np



"""# Read parameters from command line
optparser = optparse.OptionParser()
optparser.add_option(
    "-T", "--train", default="",
    help="Train set location"
)
optparser.add_option(
    "-d", "--dev", default="",
    help="Dev set location"
)
optparser.add_option(
    "-t", "--test", default="",
    help="Test set location"
)
optparser.add_option(
    "-s", "--tag_scheme", default="iobes",
    help="Tagging scheme (IOB or IOBES)"
)
optparser.add_option(
    "-l", "--lower", default="0",
    type='int', help="Lowercase words (this will not affect character inputs)"
)
optparser.add_option(
    "-z", "--zeros", default="0",
    type='int', help="Replace digits with 0"
)
optparser.add_option(
    "-c", "--char_dim", default="25",
    type='int', help="Char embedding dimension"
)
optparser.add_option(
    "-C", "--char_lstm_dim", default="25",
    type='int', help="Char LSTM hidden layer size"
)
optparser.add_option(
    "-w", "--word_dim", default="100",
    type='int', help="Token embedding dimension"
)
optparser.add_option(
    "-W", "--word_lstm_dim", default="100",
    type='int', help="Token LSTM hidden layer size"
)
optparser.add_option(
    "-p", "--pre_emb", default="",
    help="Location of pretrained embeddings"
)
optparser.add_option(
    "-A", "--all_emb", default="0",
    type='int', help="Load all embeddings"
)
optparser.add_option(
    "-a", "--cap_dim", default="0",
    type='int', help="Capitalization feature dimension (0 to disable)"
)
optparser.add_option(
    "-f", "--crf", default="1",
    type='int', help="Use CRF (0 to disable)"
)
optparser.add_option(
    "-D", "--dropout", default="0.5",
    type='float', help="Droupout on the input (0 = no dropout)"
)
optparser.add_option(
    "-L", "--lr_method", default="sgd-lr_.005",
    help="Learning method (SGD, Adadelta, Adam..)"
)
optparser.add_option(
    "-r", "--reload", default="0",
    type='int', help="Reload the last saved model"
)
opts = optparser.parse_args()[0]"""


def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)



# Parse parameters
parameters = OrderedDict()
parameters['tag_scheme'] = 'iob'
parameters['lower'] = 1 == 1
parameters['zeros'] = 1 == 1
parameters['char_dim'] = 25
parameters['char_lstm_dim'] = 25
parameters['word_dim'] = 100
parameters['word_lstm_dim'] = 100
parameters['pre_emb'] = '/Data/vectors_20.txt'
parameters['all_emb'] = 1 == 1
parameters['cap_dim'] = 0
parameters['crf'] = 1 == 1
parameters['dropout'] = 0.5
parameters['lr_method'] = "sgd-lr_.005"

train_data = '/Data/train_IBO_short_bert.txt'
test_data = '/Data/test_IBO_short_bert.txt'
dev_data = '/Data/dev_IBO_short_bert.txt'
reload = 0
# Check parameters validity
assert os.path.isfile(train_data)
assert os.path.isfile(dev_data)
assert os.path.isfile(test_data)
assert parameters['char_dim'] > 0 or parameters['word_dim'] > 0
assert 0. <= parameters['dropout'] < 1.0
assert parameters['tag_scheme'] in ['iob', 'iobes']
assert not parameters['all_emb'] or parameters['pre_emb']
assert not parameters['pre_emb'] or parameters['word_dim'] > 0
assert not parameters['pre_emb'] or os.path.isfile(parameters['pre_emb'])

# Check evaluation script / folders
if not os.path.isfile(eval_script):
    raise Exception('CoNLL evaluation script not found at "%s"' % eval_script)
if not os.path.exists(eval_temp):
    os.makedirs(eval_temp)
if not os.path.exists(models_path):
    os.makedirs(models_path)

# manual parameters
date = '201122_1'
bert= True
mask = False
mask_per = 0.05
n_epochs = 25
model_name = 'ner_short_'+date
output_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_short_'+date+'.csv')
columns = ['name', 'dataset', 'training_time', 'precision', 'recall', 'f1_score']

# Data parameters
lower = parameters['lower']
zeros = parameters['zeros']
tag_scheme = parameters['tag_scheme']

# Load sentences
train_sentences = load_sentences(train_data, zeros)
dev_sentences = load_sentences(dev_data, zeros)
test_sentences = load_sentences(test_data, zeros)

# Use selected tagging scheme (IOB / IOBES)
update_tag_scheme(train_sentences, tag_scheme)
update_tag_scheme(dev_sentences, tag_scheme)
update_tag_scheme(test_sentences, tag_scheme)

# Create a dictionary / mapping of words, chars and tags
# If we use pretrained embeddings, we add them to the dictionary.
if parameters['pre_emb']:
    embeddings_index = Word2Vec.load('w2v_clean.bin')
    dico_words_train = word_mapping(train_sentences + test_sentences + dev_sentences, lower)[0]
    dico_words, word_to_id, id_to_word = augment_with_pretrained_new(
        dico_words_train.copy(),
        embeddings_index,#parameters['pre_emb'],
        list(itertools.chain.from_iterable(
            [[w[0] for w in s] for s in dev_sentences + test_sentences])
        ) if not parameters['all_emb'] else None
    )
else:
    dico_words, word_to_id, id_to_word = word_mapping(train_sentences, lower)

# prep pre train word embeddings
if parameters['pre_emb']:
    num_words = len(id_to_word)
    embedding_matrix = np.zeros((num_words, parameters['word_dim']))
    for word, i in dico_words.items():
        if i > num_words:
            continue
        if word in embeddings_index.wv.vocab:
            embedding_matrix[i] = embeddings_index[word]






dico_chars, char_to_id, id_to_char = char_mapping(train_sentences + test_sentences + dev_sentences)
dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences + test_sentences + dev_sentences)

# save_obj(dico_words, 'dico_words_bert')
# save_obj(word_to_id, 'word_to_ids_bert')
# save_obj(id_to_word, 'id_to_words_bert')
# save_obj(dico_chars, 'dico_charss_bert')
# save_obj(char_to_id, 'char_to_ids_bert')
# save_obj(id_to_char, 'id_to_chars_bert')
# save_obj(dico_tags, 'dico_tagss_bert')
# save_obj(tag_to_id, 'tag_to_ids_bert')
# save_obj(id_to_tag, 'id_to_tags_bert')


# prepare train,dev,test datasets
train_words, train_chars, train_caps, train_tags, train_bert_words = prepare_dataset(train_sentences, parameters['word_dim'],
                                                            parameters['char_dim'], word_to_id, id_to_word, char_to_id, tag_to_id
                                                            ,lower=parameters['lower'], mask=mask, mask_per=mask_per, train=True)
dev_words, dev_chars, dev_caps, dev_tags, dev_bert_words = prepare_dataset(dev_sentences, parameters['word_dim'], parameters['char_dim']
                                                           , word_to_id, id_to_word, char_to_id, tag_to_id,lower=parameters['lower'])
test_words, test_chars, test_caps, test_tags, test_bert_words = prepare_dataset(test_sentences, parameters['word_dim'],
                                                            parameters['char_dim'], word_to_id, id_to_word, char_to_id, tag_to_id,lower=parameters['lower'])

print("the number of sentences for the traing data is : {0} ".format(len(train_bert_words)))
print("the number of sentences for the dev data is : {0} ".format(len(dev_bert_words)))
print("the number of sentences for the test data is : {0} ".format(len(test_bert_words)))
save_obj(train_bert_words, 'train_bert_words')
save_obj(dev_bert_words, 'dev_bert_words')
save_obj(test_bert_words, 'test_bert_words')

print(len(train_bert_words))
train_tags_val = train_tags
print("start loading data")
#train_bert_words = load_obj('train_bert_words')


# print("%i / %i / %i sentences in train / dev / test." % (
#     len(train_words), len(dev_words), len(test_words)))


def get_BERT_features(bert_embedding, tokens_list):
    BERT_features = []
    max_length = 100
    bert_dim = 768
    for i, tokens in enumerate(tokens_list):
        if i%100==0:
            print(i)
        curr_emb_list = []
        for j in range(0,max_length*4,max_length):
            if j > len(tokens):
                break
            sentence = Sentence(tokens)
            #sentence = Sentence(" ".join(tokens[j:min(j + max_length, len(tokens))]))
            try:
                bert_embedding.embed(sentence)
            except:
                print('Error in bert embedding for index: {}'.format(i))
            else:
                curr_emb_list += [np.array(sentence[word_bert_i].embedding) for word_bert_i in range(len(sentence))]
        padded = np.zeros((max_length * 4, bert_dim))
        if curr_emb_list:
            unpadded = np.vstack(curr_emb_list)
            padded[:unpadded.shape[0],:] = unpadded
        BERT_features.append(padded)
    return np.array(BERT_features).reshape((len(tokens_list),max_length*4,bert_dim))


print("started bert good luck")
if bert:
    bert_embedding = BertEmbeddings('bert-base-multilingual-uncased', layers='-1')
    #tokens_list = train_bert_words # your tokens list here
    bert_features_train = get_BERT_features(bert_embedding, train_bert_words)
    save_obj(bert_features_train, 'bert_features_train_short')
    print("load dev data")
    # test_bert_words = load_obj('test_bert_words')

    bert_features_test = get_BERT_features(bert_embedding, test_bert_words)
    save_obj(bert_features_test, 'bert_features_test_short')

    print("load test data")

    # dev_bert_words = load_obj('dev_bert_words')
    bert_features_dev = get_BERT_features(bert_embedding, dev_bert_words)
    save_obj(bert_features_dev, 'bert_features_dev_short')
if bert:
    # bert_features_train = load_obj('bert_features_train_short')
    # bert_features_test = load_obj('bert_features_test_short')
    # bert_features_dev = load_obj('bert_features_dev_short')

    bert_train_data= bert_features_train[:,0:100,]
    bert_test_data= bert_features_test[:,0:100,]
    bert_dev_data= bert_features_dev[:,0:100,]

# reshape the data for the fit function
train_chars = np.array(train_chars).reshape((len(train_chars), parameters['word_dim'], parameters['char_dim']))
train_caps = np.array(train_caps).reshape(len(train_caps), parameters['word_dim'])
train_tags = np.array(train_tags).reshape((len(train_tags), parameters['word_dim'], 1))

# get all mapping dictionaries len
n_words = len(id_to_word)
n_chars = len(id_to_char)
n_tags = len(id_to_tag)
print("number of tags {0}".format(n_tags))

# adjust train_tags to run with crf
if parameters["crf"]:
    train_tags = np.array([to_categorical(i, num_classes=n_tags) for i in train_tags])

directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
if not os.path.exists(directory):
    os.makedirs(directory)
models_path = directory

# Initialize model
if reload:
    if parameters['crf']:
        model = load_model(models_path + '/{}_model.h5'.format(model_name), custom_objects=create_custom_objects())
    else:
        model = load_model(models_path + '/{}_model.h5'.format(model_name))
    now = time.time()
else:
    if bert:
        model = build_bert(char_dim=parameters['char_dim'], char_lstm_dim=parameters['char_lstm_dim'],
                      word_dim=parameters['word_dim'], dropout=parameters['dropout'], n_words=n_words,
                      n_chars=n_chars, n_tags=n_tags, crf=parameters['crf'],embedding_matrix=embedding_matrix, pre_emd=parameters['pre_emb'])
        model.summary()
        now = time.time()

        model.fit([train_words, train_chars, train_caps, bert_train_data], train_tags, batch_size=32, epochs=n_epochs, validation_split=0.1,
              verbose=1,)
    else:
        model = bulid(char_dim=parameters['char_dim'], char_lstm_dim=parameters['char_lstm_dim'],
                      word_dim=parameters['word_dim'], dropout=parameters['dropout'], n_words=n_words,
                      n_chars=n_chars, n_tags=n_tags, crf=parameters['crf'],embedding_matrix=embedding_matrix, pre_emd=parameters['pre_emb'])
        model.summary()
        now = time.time()
        model.fit([train_words, train_chars, train_caps], train_tags, batch_size=32, epochs=n_epochs,
                  validation_split=0.1,
                  verbose=1, )
training_time = time.time() - now

print("Time for training model is :{} seconds".format(training_time))
if not reload:
    model.save(models_path + '/{}_model.h5'.format(model_name))
    # Save model architecture png file
    plot_model(model, to_file=models_path + '/{}_model.png'.format(model_name))
# todo change save model to save wights + save the dictionaries and mapping
# predictions
test_chars = np.array(test_chars).reshape((len(test_chars), parameters['word_dim'], parameters['char_dim']))
test_caps = np.array(test_caps).reshape(len(test_caps), parameters['word_dim'])
dev_chars = np.array(dev_chars).reshape((len(dev_chars), parameters['word_dim'], parameters['char_dim']))
dev_caps = np.array(dev_caps).reshape(len(dev_caps), parameters['word_dim'])

if bert:
    train_y_pred = model.predict([train_words, train_chars, train_caps,bert_train_data])
    dev_y_pred = model.predict([dev_words, dev_chars, dev_caps, bert_dev_data])
    y_pred = model.predict([test_words, test_chars, test_caps, bert_test_data])

else:
    train_y_pred = model.predict([train_words, train_chars, train_caps])
    dev_y_pred = model.predict([dev_words, dev_chars, dev_caps])
    y_pred = model.predict([test_words, test_chars, test_caps])
    #dev_y_pred_prob = model.predict_proba([dev_words, dev_chars, dev_caps])
    #y_pred_prob = model.predict_proba([test_words, test_chars, test_caps])
# check predictions per sentence
all_results_list = []
cur_list = evaluate(train_y_pred, id_to_word, train_words, train_tags_val, id_to_tag, model_name, training_time, 'train')
all_results_list.append(cur_list)
cur_list = evaluate(dev_y_pred, id_to_word, dev_words, dev_tags, id_to_tag, model_name, training_time, 'dev')
all_results_list.append(cur_list)
cur_list = evaluate(y_pred, id_to_word, test_words, test_tags, id_to_tag, model_name, training_time, 'test')
all_results_list.append(cur_list)
all_results_df = pd.DataFrame(all_results_list, columns=columns)
all_results_df.to_csv(output_file_path, index=False)

evaluate_auc(id_to_tag, train_y_pred, id_to_word, train_tags_val, train_words, 'train_MASK_'+date)
evaluate_auc(id_to_tag, y_pred, id_to_word, test_tags, test_words, 'test_MASK_'+date)

print("Done!!")

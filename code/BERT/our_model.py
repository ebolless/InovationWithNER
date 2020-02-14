from keras.models import Model, Input
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, TimeDistributed, concatenate, SpatialDropout1D, Masking
from keras.optimizers import SGD
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy ,crf_marginal_accuracy


def bulid(char_dim=25, char_lstm_dim=25, word_dim=100, dropout=0.5, n_words=0, n_chars=0, n_tags=0, crf=True, embedding_matrix={}, pre_emd=0):
    """
    Build the network.
    """
    input_dim = 0
    
    # 1. Embedding layer for words   
    word_in = Input(shape=(word_dim,))
    text_mask = Masking(mask_value=0., name='text_mask')(word_in)

    input_dim += word_dim
    if pre_emd:
        emb_word = Embedding(input_dim=n_words,output_dim=word_dim,input_length=word_dim,mask_zero=True, weights=[embedding_matrix], trainable=False)(text_mask)
    else:
        emb_word = Embedding(input_dim=n_words, output_dim=word_dim, input_length=word_dim, mask_zero=True)(word_in)
    # 2. Embedding layer + bidrectional LSTM - for chars
    char_in = Input(shape=(word_dim, char_dim,))
    
    emb_char = TimeDistributed(Embedding(input_dim=n_chars, output_dim=char_dim,input_length=char_dim, mask_zero=True))(char_in)
    bi_LSTM_char=TimeDistributed(Bidirectional(LSTM(units=char_lstm_dim), merge_mode='concat'))(emb_char)
    input_dim += char_lstm_dim

    # 3. Embedding layer for caps   
    caps_in = Input(shape=(word_dim,))

    input_dim += word_dim
    emb_caps = Embedding(input_dim=n_words,output_dim=word_dim,input_length=word_dim,mask_zero=True)(caps_in)


    # 4. main LSTM
    x = concatenate([emb_word, bi_LSTM_char, emb_caps])
    # x = concatenate([emb_word, bi_LSTM_char, emb_caps, bert_in])
    drop = SpatialDropout1D(dropout)(x)
    main_model = Bidirectional(LSTM(units=input_dim, return_sequences=True, dropout=dropout), merge_mode='concat')(drop)
    tan = Dense(input_dim, activation="tanh")(main_model)
    hidden = Dense(units=n_tags)(tan)
    # Decide whether we run with CRF or softmax
    if not crf:
        output = Dense(units=n_tags, activation="softmax")(hidden)
        optimizer = SGD(lr=0.01, clipnorm=5.0)
        model = Model([word_in, char_in, caps_in, bert_in], output)
        model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["acc"])
    else:
        crf = CRF(n_tags, learn_mode='marginal')
        output = crf(hidden)
        model = Model([word_in, char_in, caps_in], output)
        model.compile('rmsprop', loss=crf_loss, metrics=[crf_marginal_accuracy])#[crf_viterbi_accuracy])
    return model

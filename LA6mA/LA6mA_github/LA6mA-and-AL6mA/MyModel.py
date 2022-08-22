from keras.layers import Multiply, Bidirectional, Dropout
from keras.layers import LSTM
from keras.models import  Model
from keras import Input
from keras.layers.core import Permute, Reshape, Dense, Lambda,  RepeatVector, Flatten
from keras import backend as K

SINGLE_ATTENTION_VECTOR = False
INPUT_DIM = 4
TIME_STEPS = 41


def attention_3d_block(inputs):
    # inputs = [41,32]
    input_dim = int(inputs.shape[2])
    
    a = Permute((2, 1))(inputs)
    # a = [32,41]
    a = Reshape((input_dim, TIME_STEPS))(a)
    # a = [32,41] = MT
    a = Dense(TIME_STEPS, activation='softmax')(a)
    # a = T ,[32,41]
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    # a_probs = [41,32]
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul


# LA6mA
def LA6mA():
    K.clear_session()
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    lstm_units = 32
    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True), merge_mode='sum')(inputs)
    lstm_out = Dropout(0.2)(lstm_out)
    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True), merge_mode='sum')(lstm_out)
    lstm_out = Dropout(0.2)(lstm_out)
    attention_mul = attention_3d_block(lstm_out)
    attention_mul = Flatten()(attention_mul)
    layer = Dense(100)(attention_mul)
    output = Dense(1, activation='sigmoid')(layer)
    model = Model(inputs=[inputs], outputs=output,name="LA6mA")
    return model


# LA6mA without attention
def LA6mA_al():
    K.clear_session()
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    lstm_units = 32
    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True), merge_mode='sum')(inputs)
    lstm_out = Dropout(0.2)(lstm_out)
    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True), merge_mode='sum')(lstm_out)
    lstm_out = Dropout(0.2)(lstm_out)
    lstm_out = Flatten()(lstm_out)
    layer = Dense(100)(lstm_out)
    output = Dense(1, activation='sigmoid')(layer)
    model = Model(inputs=[inputs], outputs=output,name="LA6mA_al")
    return model


# AL6mA
def AL6mA():
    K.clear_session()
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    attention_mul = attention_3d_block(inputs)
    lstm_units = 128
    # attention_mul = LSTM(lstm_units, return_sequences=False)(attention_mul)
    attention_mul = Bidirectional(LSTM(lstm_units, return_sequences=False), merge_mode='sum')(attention_mul)
    attention_mul = Dropout(0.2)(attention_mul)
    # attention_mul = Flatten()(attention_mul)
    # attention_mul = Dense(100)(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(inputs=[inputs], outputs=output,name="AL6mA")
    return model


# AL6mA without attention
def AL6mA_al():
    K.clear_session()
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    attention_mul = attention_3d_block(inputs)
    lstm_units = 128
    attention_mul = Bidirectional(LSTM(lstm_units, return_sequences=False), merge_mode='sum')(attention_mul)
    attention_mul = Dropout(0.2)(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(inputs=[inputs], outputs=output,name="AL6mA_AL")
    return model

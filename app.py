import numpy as np
import os
from train_data_generator import get_train_data, load_csv
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Input, BatchNormalization, MaxPool1D, Conv2D, Conv1D
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
def new_model():
    # encoder
    encoder_inputs = Input(shape=(14,8))
    encoder_outputs = LSTM(16, return_sequences = True)(encoder_inputs)
    encoder_outputs = BatchNormalization()(encoder_outputs)
    encoder_outputs= LSTM(32, return_sequences = True)(encoder_outputs)
    encoder_outputs = BatchNormalization()(encoder_outputs)
    encoder_outputs_and_states = LSTM(16, return_state=True, return_sequences = False)(encoder_outputs)
    encoder_states = encoder_outputs_and_states[1:]
    decoder_inputs = Input(shape=(None,7))
    decoder_output = LSTM(16, return_sequences = True)(decoder_inputs, initial_state=encoder_states)
    decoder_output = BatchNormalization()(decoder_output)
    output = Dense(1)(decoder_output)
    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=output)
    model.summary()
    return model

model = new_model()
model.compile(loss='mse',
              optimizer='adam',)
model.summary()
# data preprocess
train_days = 14
pred_days = 7
train_col_idx = [0, 2, 5, 6]

train_data = load_csv('2016-2017data.csv', train_col_idx)
test_data = load_csv('2018data.csv', train_col_idx)[1:]
train_x, train_y, train_x2 =  get_train_data(train_data, pred_days, train_days)
test_x, test_y, test_x2 = get_train_data(test_data, pred_days, train_days)
np.save('train_x.npy', train_x)
np.save('train_y.npy', train_y)
np.save('train_x2.npy', train_x2)
np.save('test_x.npy', test_x)
np.save('test_y.npy', test_y)
np.save('test_x2.npy', test_x2)


train_x = np.load('train_x.npy')
train_y = np.load('train_y.npy')
train_x2 = np.load('train_x2.npy')
test_x = np.load('test_x.npy')
test_y = np.load('test_y.npy')
test_x2 = np.load('test_x2.npy')
train_x = np.concatenate((train_x, test_x), axis = 0)
train_y = np.concatenate((train_y, test_y), axis = 0)
train_x2 = np.concatenate((train_x2, test_x2), axis = 0)
model.fit([train_x[::,::,[0,3,4,5,6,7,8,9]], train_x2], train_y[::,::,np.newaxis], epochs=100, batch_size=8)
test_data = np.array([[
                    [2.8338,1,0,0,0,0,0,0],#3/18
                    [2.8882,0,1,0,0,0,0,0],#3/19
                    [2.9645,0,0,1,0,0,0,0],#3/20
                    [3.0343,0,0,0,1,0,0,0],#3/21
                    [2.9618,0,0,0,0,1,0,0],#3/22
                    [2.5265,0,0,0,0,0,1,0],#3/23
                    [2.4812,0,0,0,0,0,0,1],#3/24
                    [2.8535,1,0,0,0,0,0,0],#3/25
                    [2.8756,0,1,0,0,0,0,0],#3/26
                    [2.9140,0,0,1,0,0,0,0],#3/27
                    [3.0093,0,0,0,1,0,0,0],#3/28
                    [2.9673,0,0,0,0,1,0,0],#3/29
                    [2.5810,0,0,0,0,0,1,0],#3/30
                    [2.4466,0,0,0,0,0,0,1]#3/31
                    ]])
test_data2 = np.array([[
            [1,0,0,0,0,0,0],#4/1
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1],
            [1,0,0,0,0,0,0]
            ]])
pred = model.predict([test_data, test_data2])
print(pred)
date = [20190402,20190403,20190404,20190405,20190406,20190407,20190408]
with open('submission.csv', 'w') as f:
    f.write('date,peak_load(MW)')
    for i in range(7):
        f.write('\n%d,%5d' % (date[i], pred[0,i+1,0]*10000))





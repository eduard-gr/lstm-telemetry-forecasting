from sys import argv
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import set_option
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import RNN
import tensorflow as tf
from tensorflow import keras


if(len(argv) != 5):
    print("unit epoch batch")
    print("telemetry-lstm.py LSTM 100 100 70")
    exit(1)

print(tf.__version__)

_, rnn, unit, epoch, batch = argv

rnn = rnn.upper()

if(rnn not in ['RNN', 'LSTM', 'GRU']):
	print('unknown learning algorithm',rnn)
	exit(1)

print(
	'rnn:', rnn,
	'unit:', unit,
	'epoch:', epoch,
	'batch:', batch)


# Set CPU as available physical device
#my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
#tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')

def series_to_supervised(data):
    n_vars = 1 if type(data) is list else data.shape[1]

    #df = DataFrame(data)
    cols, names = list(), list()

    cols.append(data.shift(1))
    names += [('var%d(t-%d)' % (j + 1, 1)) for j in range(n_vars)]

    cols.append(data.shift(-0)[0])
    names += ['var1(1)']

    agg = concat(cols, axis=1)
    agg.columns = names

    return agg

# load dataset
df = read_csv('fm_volvo_odb_telemetry.csv', index_col=0).interpolate(method='linear', axis=0)

values = df.values

# ensure all data is float
values = values.astype('float32')

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = DataFrame(scaler.fit_transform(values))

print(scaled.head())

reframed = series_to_supervised(scaled)

#Udaljaem NaN
reframed.dropna(inplace=True)

#print("supervised series")
#print(reframed.head())

test_size = 1000
test = reframed.iloc[:test_size, :]
train = reframed.iloc[test_size:, :]

#print('train:', len(train.index))
# print(train.head())
# print('...')
# print(train.tail())

#print('test:', len(test.index))
# print(test.head())
# print('...')
# print(test.tail())


# Otdeljaem zpolsednjuju kolonku v kotrooj hranatjsa znachnija rpm + 1
train_X, train_y = train.values[:, :-1], train.values[:, -1]
test_X, test_y = test.values[:, :-1], test.values[:, -1]


# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

#print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

#design network
model = Sequential()

if (rnn == "RNN"):
    model.add(RNN(unit, input_shape=(train_X.shape[1], train_X.shape[2])))
elif (rnn == "LSTM"):
    model.add(LSTM(unit, input_shape=(train_X.shape[1], train_X.shape[2])))
elif (rnn == "GRU"):
    model.add(GRU(unit, input_shape=(train_X.shape[1], train_X.shape[2])))
else:
    exit(1)

model.add(Dense(1))

model.compile(loss='mae', optimizer='adam')

# fit network
history = model.fit(
    train_X,
    train_y,
    epochs=epoch,
    batch_size=batch,
    validation_data=(test_X, test_y),
    verbose=1,
    shuffle=False)

# print("save model")
model.save('model_' + rnn + '_' + unit + '_' + epoch + '_' + batch)

#model = keras.models.load_model('telemetry-model')

# plot history
pyplot.plot(history.history['loss'], label='loss')
pyplot.plot(history.history['val_loss'], label='val loss')
pyplot.plot(history.history['accuracy'], label='accuracy')
pyplot.plot(history.history['val_accuracy'], label='val_accuracy')
pyplot.legend()
pyplot.title('Fit rnn:%s unit:%d epoch:%d batch:%d ' % (rnn, unit, epoch, batch))
pyplot.savefig('fit_' + rnn + '_' + unit + '_' + epoch + '_' + batch + '.png')

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)

inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]


# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))

inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))

pyplot.figure()
pyplot.plot(inv_y, label='RPM')
pyplot.plot(inv_yhat, label='Predicted')
pyplot.legend()
pyplot.title('Prediction result RMSE:%.3f rnn:%s unit:%d epoch:%d batch:%d ' % (rmse, rnn, unit, epoch, batch))
pyplot.savefig('predict_' + rnn + '_' + unit + '_' + epoch + '_' + batch + '.png')

#virtualenv -p python3.7 ./venv
#source ./venv/bin/activate
#pip install --upgrade pip
#pip install matplotlib tensorflow pandas sklearn keras
#https://medium.com/@alexmarginean/installing-tensorflow-on-fedora-29-862573ef2ab9
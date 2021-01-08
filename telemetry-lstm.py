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
import tensorflow as tf
from tensorflow import keras

print(tf.__version__)


#set_option('display.max_columns', None)

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
#, header=0, index_col=0
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

print("supervised series")
print(reframed.head())

test_size = 1000
test = reframed.iloc[:test_size, :]
train = reframed.iloc[test_size:, :]

print('train:', len(train.index))
# print(train.head())
# print('...')
# print(train.tail())

print('test:', len(test.index))
# print(test.head())
# print('...')
# print(test.tail())


# Otdeljaem zpolsednjuju kolonku v kotrooj hranatjsa znachnija rpm + 1
train_X, train_y = train.values[:, :-1], train.values[:, -1]
test_X, test_y = test.values[:, :-1], test.values[:, -1]

#print('train_y')
#print(train_y)

#print('test_y')
#print(test_y)

#exit(0)
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

print(
    train_X.shape,
    train_y.shape,
    test_X.shape,
    test_y.shape)

# print('train_X&Y')
# print(train_X)
# print(train_y)
#
# print('test_X&Y')
# print(test_X)
# print(test_y)

#design network
model = Sequential()
model.add(GRU(100, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))

model.compile(
    loss='mae',
    optimizer='adam')

# fit network
history = model.fit(
    train_X,
    train_y,
    epochs=100,
    batch_size=72,
    validation_data=(test_X, test_y),
    verbose=1,
    shuffle=False)

# print("save model")
# model.save('telemetry-model')

#model = keras.models.load_model('telemetry-model')

# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
#pyplot.show()
pyplot.savefig('test.png')

# make a prediction
yhat = model.predict(test_X)
#yhat[yhat < 0] = 0
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


# print("predicted")
# df_yhat = DataFrame(yhat)
# print(df_yhat.head())
# print("...")
# print(df_yhat.tail())


# print("inv_y")
# df_test_y = DataFrame(test_y)
# print(df_test_y.head())
# print("...")
# print(df_test_y.tail())

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

pyplot.figure()
#pyplot.subplots_adjust(hspace=0.5,bottom=0.05,top=0.95)

pyplot.subplot(1, 1, 1)
pyplot.plot(inv_y)
pyplot.title("RPM", y=1, loc='left')

pyplot.plot(inv_yhat)
pyplot.title("Predicted", y=1, loc='left')
pyplot.savefig('test2.png')

#virtualenv -p python3.7 ./venv
#source ./venv/bin/activate
#pip install --upgrade pip
#pip install matplotlib tensorflow pandas sklearn keras
#https://medium.com/@alexmarginean/installing-tensorflow-on-fedora-29-862573ef2ab9
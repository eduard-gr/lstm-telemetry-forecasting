import pandas as pd
import matplotlib.pyplot as pyplot
import mplcursors
import seaborn as sns

test_size = 1000
df = pd.read_csv('fm_volvo_odb_telemetry.json').iloc[:test_size, :]

#read_csv('recent-grads.csv')
#print(dt.shape)
#print(dt.dtypes)

#print(df.head(3))

#print()


#print(recent_grads.sample(5, random_state=0))

#recent_grads['Temp'].plot(linewidth=0.5);


#dt.plot(y=["EngineLoad", "VehicleSpeed","ThrottlePosition"])
#dt.plot(y=["CoolantTemperature", "IntakeAirTemperature", "AmbientAirTemperature"])

#dt.plot(y=["EngineRPM", "MAF","ThrottlePosition"])

#df.plot(y=["DirectFuelRailPressure","FuelLevel"])
#pyplot.show()

#{"DTC": 0, "MAF": 2958, "Date": "2020-11-22T16:38:48", "EGRError": 0, "Ignition": 1, "EngineRPM": 1405, "FuelLevel": 99, "IntakeMAP": 113, "EngineLoad": 47, "CommandedEGR": 15, "VehicleSpeed": 76, "ThrottlePosition": 7,
# "BarometicPressure": 100, "CoolantTemperature": 88, "ControlModuleVoltage": 14960, "IntakeAirTemperature": 8, "AmbientAirTemperature": 5, "DirectFuelRailPressure": 9478},

plots = 14

pyplot.figure()
pyplot.subplots_adjust(hspace=0.5,bottom=0.05,top=0.95)


#pyplot.plot(df["MAF"].values)
#pyplot.title("MAF", y=1, loc='right')
def position_cenerator():
    i = 0
    while True:
        i += 1
        yield i

position = position_cenerator()


pyplot.subplot(plots, 1, next(position))
pyplot.plot(df["speed"].interpolate(method='akima', order=5).values)
pyplot.title("VehicleSpeed", y=1, loc='left')

pyplot.subplot(plots, 1, next(position))
pyplot.plot(df["maf"].interpolate(method='akima', order=5).values)
pyplot.title("MAF", y=1, loc='left')

pyplot.subplot(plots, 1, next(position))
pyplot.plot(df["engine_rpm"].interpolate(method='akima', order=5).values)
pyplot.title("EngineRPM", y=1, loc='left')

pyplot.subplot(plots, 1, next(position))
pyplot.plot(df["egr_error"].interpolate(method='akima', order=5).values)
pyplot.title("EGRError", y=1, loc='left')

pyplot.subplot(plots, 1, next(position))
pyplot.plot(df["commanded_egr"].interpolate(method='akima', order=5).values)
pyplot.title("CommandedEGR", y=1, loc='left')

pyplot.subplot(plots, 1, next(position))
pyplot.plot(df["intake_map"].interpolate(method='akima', order=5).values)
pyplot.title("IntakeMAP", y=1, loc='left')

pyplot.subplot(plots, 1, next(position))
pyplot.plot(df["engine_load"].interpolate(method='akima', order=5).values)
pyplot.title("EngineLoad", y=1, loc='left')


pyplot.subplot(plots, 1, next(position))
pyplot.plot(df["throttle_position"].interpolate(method='akima', order=5).values)
pyplot.title("ThrottlePosition", y=1, loc='left')

pyplot.subplot(plots, 1, next(position))
pyplot.plot(df["barometic_pressure"].interpolate(method='akima', order=5).values)
pyplot.title("BarometicPressure", y=1, loc='left')

pyplot.subplot(plots, 1, next(position))
pyplot.plot(df["direct_fuel_rail_pressure"].interpolate(method='akima', order=5).values)
pyplot.title("DirectFuelRailPressure", y=1, loc='left')

pyplot.subplot(plots, 1, next(position))
pyplot.plot(df["ignition"].values)
pyplot.plot(df["movement"].values)
pyplot.title("I\M", y=1, loc='left')

pyplot.subplot(plots, 1, next(position))
pyplot.plot(df["fuel_level"].values)
pyplot.title("FuelLevel", y=1, loc='left')

pyplot.subplot(plots, 1, next(position))
pyplot.plot(df["coolant_temperature"].values)
pyplot.title("CoolantTemperature", y=1, loc='left')

#pyplot.subplot(plots, 1, 14)
pyplot.plot(df["intake_air_temperature"].values)
pyplot.title("IntakeAirTemperature", y=1, loc='left')

#pyplot.subplot(plots, 1, 15)
pyplot.plot(df["ambient_air_temperature"].values)
pyplot.title("AmbientAirTemperature", y=1, loc='left')

pyplot.subplot(plots, 1, next(position))
pyplot.plot(df["control_module_voltage"].values)
pyplot.title("ControlModuleVoltage", y=1, loc='left')

#mplcursors.cursor(hover=True)
pyplot.show()



#https://www.dataquest.io/blog/tutorial-time-series-analysis-with-pandas/
#https://towardsdatascience.com/machine-learning-recurrent-neural-networks-and-long-short-term-memory-lstm-python-keras-example-86001ceaaebc
#https://www.tensorflow.org/guide/keras/rnn
#https://adventuresinmachinelearning.com/keras-lstm-tutorial/
#https://keras.io/examples/
#https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
#https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/



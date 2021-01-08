import pandas as pd
import matplotlib.pyplot as pyplot
import mplcursors
import seaborn as sns

test_size = 1000

df = pd.read_csv('fm_volvo_odb_telemetry.csv', index_col=0).iloc[:test_size, :]
#.interpolate(method='linear', axis=0)
#df.info()

#read_csv('recent-grads.csv')
#print(dt.shape)
#print(dt.dtypes)

#print(df.head())

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

plots = 4 #14

pyplot.figure()
pyplot.subplots_adjust(hspace=0.5,bottom=0.05,top=0.95)
#pyplot.title("EngineRPM", y=1, loc='left')


#pyplot.plot(df["MAF"].values)
#pyplot.title("MAF", y=1, loc='right')
def position_cenerator():
    i = 0
    while True:
        i += 1
        yield i

position = position_cenerator()



#.interpolate(method='akima', order=5)

pyplot.subplot(plots, 1, next(position))
pyplot.plot(df["speed"].values)
pyplot.ylabel('Speed')
#pyplot.legend()
pyplot.title('Prediction result RMSE:%.3f Test:%d ' % (0.543534543423, 123))
# pyplot.subplot(plots, 1, next(position))
# pyplot.plot(df["maf"].values)
# pyplot.title("MAF", y=1, loc='left')

pyplot.subplot(plots, 1, next(position))
pyplot.plot(df["engine_rpm"].values, label='RPM')
pyplot.ylabel('RPM')
#pyplot.legend()




# pyplot.subplot(plots, 1, next(position))
# pyplot.plot(df["throttle_position"].values)
# pyplot.title("ThrottlePosition", y=1, loc='left')

pyplot.subplot(plots, 1, next(position))
pyplot.plot(df["direct_fuel_rail_pressure"].values, label='DirectFuelRailPressure')
pyplot.ylabel('DirectFuelRailPressure')
#pyplot.legend()

pyplot.subplot(plots, 1, next(position))
pyplot.plot(df["fuel_level"].values, label='FuelLevel')
pyplot.ylabel('FuelLevel')
#pyplot.legend()

#pyplot.title("FuelLevel", y=1, loc='left')

# pyplot.subplot(plots, 1, next(position))
# pyplot.plot(df["egr_error"].values)
# pyplot.title("EGRError", y=1, loc='left')

# pyplot.subplot(plots, 1, next(position))
# pyplot.plot(df["commanded_egr"].values)
# pyplot.title("CommandedEGR", y=1, loc='left')

# pyplot.subplot(plots, 1, next(position))
# pyplot.plot(df["intake_map"].values)
# pyplot.title("IntakeMAP", y=1, loc='left')

# pyplot.subplot(plots, 1, next(position))
# pyplot.plot(df["engine_load"].values)
# pyplot.title("EngineLoad", y=1, loc='left')



# pyplot.subplot(plots, 1, next(position))
# pyplot.plot(df["barometic_pressure"].values)
# pyplot.title("BarometicPressure", y=1, loc='left')



# pyplot.subplot(plots, 1, next(position))
# pyplot.plot(df["fmmu_type"].values)
# #pyplot.plot(df["movement"].values)
# pyplot.title("I\M", y=1, loc='left')


# pyplot.subplot(plots, 1, next(position))
# pyplot.plot(df["coolant_temperature"].values)
# pyplot.title("CoolantTemperature", y=1, loc='left')

# #pyplot.subplot(plots, 1, 14)
# pyplot.plot(df["intake_air_temperature"].values)
# pyplot.title("IntakeAirTemperature", y=1, loc='left')

# #pyplot.subplot(plots, 1, 15)
# pyplot.plot(df["ambient_air_temperature"].values)
# pyplot.title("AmbientAirTemperature", y=1, loc='left')

# pyplot.subplot(plots, 1, next(position))
# pyplot.plot(df["control_module_voltage"].values)
# pyplot.title("ControlModuleVoltage", y=1, loc='left')

#mplcursors.cursor(hover=True)

pyplot.show()



#https://www.dataquest.io/blog/tutorial-time-series-analysis-with-pandas/
#https://towardsdatascience.com/machine-learning-recurrent-neural-networks-and-long-short-term-memory-lstm-python-keras-example-86001ceaaebc
#https://www.tensorflow.org/guide/keras/rnn
#https://adventuresinmachinelearning.com/keras-lstm-tutorial/
#https://keras.io/examples/
#https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
#https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/



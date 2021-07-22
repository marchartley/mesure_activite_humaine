# -*- coding: utf-8 -*-


from xiaopir import XIAOPIR
import time
import pandas as pd
import datetime
import influxdb as ifdb
#import paho.mqtt.client as mqtt
#import socket
import matplotlib.pyplot as plt


pir0 = XIAOPIR(0x20)
pir1 = XIAOPIR(0x21)
pir2 = XIAOPIR(0x22)
pir3 = XIAOPIR(0x23)

PIR_FREQUENCY = 40
SEQUENCE_DURATION = 20.0 # Number of seconds
    
def start_experiment():
    curtime = datetime.datetime.utcnow() + pd.Timedelta(1000/PIR_FREQUENCY,'ms')
    pir0.enableMeasurement()
    pir1.enableMeasurement()
    pir2.enableMeasurement()
    pir3.enableMeasurement()

    time.sleep(.1)
    # This Data Frame will store all our values at the end
    dfdata = pd.DataFrame(columns=['timestamp','pir0','pir1','pir2','pir3'])
    
    # To be faster in time, store all the values in arrays
    # Then put the arrays in the dataframe at once
    data_0 = []
    data_1 = []
    data_2 = []
    data_3 = []
    timestamps = []
    for i in range(int(PIR_FREQUENCY * SEQUENCE_DURATION)):
        time_begin = time.time()
        try:
            data0 = None
            data1 = None
            data2 = None
            data3 = None
            # Try the to take the measure twice to lower the error rate
            try:
                data0 = pir0.getVal()
            except:
                data0 = pir0.getVal()
                
            try:
                data1 = pir1.getVal()
            except:
                data1 = pir1.getVal()
                
            try:
                data2 = pir2.getVal()
            except:
                data2 = pir2.getVal()
                
            try:
                data3 = pir3.getVal()
            except:
                data3 = pir3.getVal()
            timestamps.append(curtime)
            
            data_0.append(data0)
            data_1.append(data1)
            data_2.append(data2)
            data_3.append(data3)
        except IOError:
            pass
        except Exception as e:
            print('error found', e)
#            raise Exception('')
            # print("error:", e)
        curtime = curtime + pd.Timedelta(1000/PIR_FREQUENCY,'ms')
        #time.sleep(1/PIR_FREQUENCY)
        # Sleep 0.1ms until exactly 1/freq seconds has past
        while time.time() < time_begin + 1/PIR_FREQUENCY:
            time.sleep(0.0001)
        
    dfdata['timestamp'] = timestamps
    dfdata['pir0'] = data_0
    dfdata['pir1'] = data_1
    dfdata['pir2'] = data_2
    dfdata['pir3'] = data_3

    dfdata = dfdata.set_index('timestamp')
#    
#    print(dfdata.describe())
#    
#    plt.plot(dfdata['pir0'], label='pir0')
#    plt.plot(dfdata['pir1'], label='pir1')
#    plt.plot(dfdata['pir2'], label='pir2')
#    plt.plot(dfdata['pir3'], label='pir3')
#    plt.legend()
#    plt.show()
#    
#    dfdata = dfdata.rolling(3,center=True).median().iloc[1:-1]
    dfdata = dfdata.interpolate(axis = 0).dropna()
    dfdata['pir0'] = dfdata['pir0'].astype(int).astype(object)
    dfdata['pir1'] = dfdata['pir1'].astype(int).astype(object)
    dfdata['pir2'] = dfdata['pir2'].astype(int).astype(object)
    dfdata['pir3'] = dfdata['pir3'].astype(int).astype(object)

#    # It is expected to loose 2 data point (first and last) because of the sliding window
#    expected = int(PIR_FREQUENCY * SEQUENCE_DURATION )
#    missing = expected - dfdata.shape[0]
#    # We expect to loose aboute 0.05% of data
#    print(f"Expecting {expected} values, missing {missing}")
#    print(f"Lost: {round(100 * (missing / expected), 2)}% of data")
    
#    plt.plot(dfdata['pir0'], label='pir0')
#    plt.plot(dfdata['pir1'], label='pir1')
#    plt.plot(dfdata['pir2'], label='pir2')
#    plt.plot(dfdata['pir3'], label='pir3')
#    plt.legend()
#    plt.show()
    clientdf.write_points(dfdata,'pir_network',
                          tags={ 'exp':'test', 
                                'move': 'on', 
                                'sensor_height': 'ankles'},
                        time_precision='ms')

# Create the client to connect to InfluxDB
clientdf = ifdb.DataFrameClient(host='192.168.143.29',port=8086,username='listic',password='groupefiad')
clientdf.switch_database('iot')
while True:
    start_time = time.time()
    start_experiment()
    elapsed_time = time.time() - start_time
    # With the PIR initialisation, the storage of the values and
    # the storage in InfluxDB, we expect to have about 
    # 500ms of latency
    # For a sequence of 5sec, that makes ~ 10% of time lost
    # For a sequence of 30sec, it's about 1.6% of time
    print(f"Time: {round(elapsed_time, 2)}s (+{round(100 * (elapsed_time - SEQUENCE_DURATION)/SEQUENCE_DURATION, 2)}%)")
clientdf.close()

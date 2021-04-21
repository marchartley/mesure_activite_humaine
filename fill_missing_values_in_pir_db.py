import datetime

import dateutil
import pandas as pd

import InfluxDB_connection as db


def join_pir_and_params(all_queries):
    all_data_joint = []
    last_pir_timestamp = None
    last_pir_values = None
    last_walking_speed_timestamp = None
    last_walking_speed = None
    sequence_id = -1
    for all_data in all_queries:
        for data in all_data:
            current_timestamp = dateutil.parser.isoparse(data['time'].replace('Z', '+00:00')).timestamp() * 1000
            if data['pir0'] is not None:
                last_pir_values = data
                last_pir_timestamp = current_timestamp
            if data['walking_speed'] is not None:
                last_walking_speed_timestamp = current_timestamp
                if last_walking_speed is None:
                    last_walking_speed = data['walking_speed']
                else:
                    last_walking_speed = (data['walking_speed'] + last_walking_speed)/2

            if last_pir_values is not None and last_walking_speed is not None and (current_timestamp - last_walking_speed_timestamp) < 1000:
                this_data = last_pir_values
                this_data['walking_speed'] = last_walking_speed
                if len(all_data_joint) == 0 or (current_timestamp - dateutil.parser.isoparse(all_data_joint[-1]['time'].replace('Z', '+00:00')).timestamp() * 1000) > 100:
                    # if len(all_data_joint) > 0:
                    #     print(f"{sequence_id} -> {sequence_id+1} : {(current_timestamp - dateutil.parser.isoparse(all_data_joint[-1]['time'].replace('Z', '+00:00')).timestamp() * 1000)}")
                    sequence_id += 1
                this_data['sequence_id'] = sequence_id
                if len(all_data_joint) == 0 or this_data['time'] != all_data_joint[-1]['time']:
                    all_data_joint.append(this_data)
    return all_data_joint


client = db.ListicClientDB(server_side=True)
all_queries = client.get_all_pir_values()
df = pd.DataFrame(join_pir_and_params(all_queries))
for i in range(df['sequence_id'].max() + 1):
    if len(df[df['sequence_id'] == i]) <= 1:
        df = df[df['sequence_id'] != i]
print(df)
df.to_csv("all_pir_values.csv")
# for i in range(30):
#     print(join_pir_and_params(all_queries)[i])


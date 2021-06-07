import pickle

import matplotlib.pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

data = pd.read_csv('sequences_resumed.csv')
data = data[data['walking_speed'] < 2.0]

plt.subplot(1, 3, 1)
plt.title("Walking speed")
plt.hist(data['walking_speed'])

plt.subplot(1, 3, 2)
plt.title('Corridor used')
plt.hist(data['corridor'], bins = 3)
plt.xticks([-1, 0, 1], ['East', 'Center', 'West'])

plt.subplot(1, 3, 3)
plt.title('Direction')
plt.hist(data['direction'], bins = 3)
plt.xticks([-1, 0, 1], ['South-North', 'Error', 'North-South'])
plt.show()

# im_label_1 = 'ENS'
# im_label_2 = 'ESN'
#
# im1 = np.array(ImageOps.grayscale(Image.open('pir_images/' + im_label_1 + '.png')))
# im2 = np.array(ImageOps.grayscale(Image.open('pir_images/' + im_label_2 + '.png')))
#
# plt.title(f"Difference between {im_label_1} and {im_label_2}")
# diff = abs(np.array(im2 - im1))
# # mini, maxi = diff.min(), diff.max()
# # diff = (diff + mini) / (maxi + mini)
# plt.imshow(diff/diff.max(), cmap='gray')
# plt.axis('off')
# plt.show()
# exit()

# all_data = pickle.load(open('all_pirs_transformed.data', 'rb'))
#
# mean_data = {
#     'ENS': {'pir0': None, 'pir1': None, 'pir2': None, 'pir3': None, 'count': 0},
#     'ONS': {'pir0': None, 'pir1': None, 'pir2': None, 'pir3': None, 'count': 0},
#     'CNS': {'pir0': None, 'pir1': None, 'pir2': None, 'pir3': None, 'count': 0},
#
#     'ESN': {'pir0': None, 'pir1': None, 'pir2': None, 'pir3': None, 'count': 0},
#     'OSN': {'pir0': None, 'pir1': None, 'pir2': None, 'pir3': None, 'count': 0},
#     'CSN': {'pir0': None, 'pir1': None, 'pir2': None, 'pir3': None, 'count': 0},
# }
#
# for data in all_data:
#     pir0, pir1, pir2, pir3, id_seq, speed, corridor, direction = data
#     corridor = round(corridor)
#     direction = round(direction)
#     category = ''
#     if direction == 0:
#         continue
#     elif corridor == -1 and direction == -1:
#         category = 'ESN'
#     elif corridor == -1 and direction == 1:
#         category = 'ENS'
#     elif corridor == 0 and direction == -1:
#         category = 'CSN'
#     elif corridor == 0 and direction == 1:
#         category = 'CNS'
#     elif corridor == 1 and direction == -1:
#         category = 'OSN'
#     elif corridor == 1 and direction == 1:
#         category = 'ONS'
#     else:
#         print(f"Corridor = {corridor}, Direction = {direction}")
#         continue
#
#     if mean_data[category]['pir0'] is None:
#         mean_data[category]['pir0'] = np.array(pir0)
#         mean_data[category]['pir1'] = np.array(pir1)
#         mean_data[category]['pir2'] = np.array(pir2)
#         mean_data[category]['pir3'] = np.array(pir3)
#     else:
#         mean_data[category]['pir0'] += np.asarray(pir0)
#         mean_data[category]['pir1'] += np.asarray(pir1)
#         mean_data[category]['pir2'] += np.asarray(pir2)
#         mean_data[category]['pir3'] += np.asarray(pir3)
#     mean_data[category]['count'] += 1
#
# for category in mean_data:
#     try:
#         mean_data[category]['pir0'] /= mean_data[category]['count']
#         mean_data[category]['pir1'] /= mean_data[category]['count']
#         mean_data[category]['pir2'] /= mean_data[category]['count']
#         mean_data[category]['pir3'] /= mean_data[category]['count']
#     except:
#         print("Error on", category)
#
#
# for category in mean_data:
#     try:
#         plt.suptitle(f"Category: {category}")
#         for pir in range(4):
#             plt.subplot(2, 2, pir+1)
#             plt.title(f"PIR {pir}")
#             plt.imshow(mean_data[category][f"pir{pir}"], cmap='gray')
#             plt.axis('off')
#         plt.savefig('pir_images/' + category + '.png', )
#         # plt.show()
#     except:
#         print(category)

# for i in range(3):
#     plt.suptitle(f"Speed: {round(data[i][5] * 3.6, 2)} km/h\n"
#                  f"Corridor {'E' if data[i][6] == -1 else 'O' if data[i][6] == 1 else 'C'}\n"
#                  f"Direction {'NS' if data[i][7] == 1 else 'SN' if data[i][7] == -1 else 'undefined'}")
#     for pir in range(4):
#         plt.subplot(2, 2, pir+1)
#         plt.title(f"PIR {pir}")
#         plt.imshow(data[i][pir])
#         plt.axis('off')
#     plt.show()

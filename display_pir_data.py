import pickle

import matplotlib.gridspec
import matplotlib.pyplot as plt
import numpy as np
from profilestats import profile
import pandas as pd
from PIL import Image, ImageOps

# data = pd.read_csv('sequences_resumed.csv')
# data = data[data['walking_speed'] < 2.0]
#
# plt.subplot(1, 3, 1)
# plt.title("Walking speed")
# plt.hist(data['walking_speed'])
#
# plt.subplot(1, 3, 2)
# plt.title('Corridor used')
# plt.hist(data['corridor'], bins = 3)
# plt.xticks([-1, 0, 1], ['East', 'Center', 'West'])
#
# plt.subplot(1, 3, 3)
# plt.title('Direction')
# plt.hist(data['direction'], bins = 3)
# plt.xticks([-1, 0, 1], ['South-North', 'Error', 'North-South'])
# plt.show()



# im_label_1 = 'ENS'
# im_label_2 = 'CSN'
#
# im1 = np.array(ImageOps.grayscale(Image.open('pir_images/' + im_label_1 + '.png')))
# im2 = np.array(ImageOps.grayscale(Image.open('pir_images/' + im_label_2 + '.png')))
#
# plt.title(f"Difference between {im_label_1} and {im_label_2}")
# diff = np.array(im2 - im1)
# # mini, maxi = diff.min(), diff.max()
# # diff = (diff + mini) / (maxi + mini)
# plt.imshow(diff/diff.max(), cmap='gray')
# plt.axis('off')
# plt.show()
# exit()

all_data = pickle.load(open('all_pirs_transformed_not_scaled.data', 'rb'))

mean_data = {
    'ENS': {'pir0': None, 'pir1': None, 'pir2': None, 'pir3': None, 'count': 0},
    'ONS': {'pir0': None, 'pir1': None, 'pir2': None, 'pir3': None, 'count': 0},
    'CNS': {'pir0': None, 'pir1': None, 'pir2': None, 'pir3': None, 'count': 0},

    'ESN': {'pir0': None, 'pir1': None, 'pir2': None, 'pir3': None, 'count': 0},
    'OSN': {'pir0': None, 'pir1': None, 'pir2': None, 'pir3': None, 'count': 0},
    'CSN': {'pir0': None, 'pir1': None, 'pir2': None, 'pir3': None, 'count': 0},
}

for i, data in enumerate(all_data):
    print(f"{round(100 * i / len(all_data), 1)}% ({i} / {len(all_data)})")

    pir0, pir1, pir2, pir3, id_seq, speed, corridor, direction, raw = data
    corridor = round(corridor)
    direction = round(direction)
    category = ''
    if direction == 0:
        continue
    elif corridor == -1 and direction == -1:
        category = 'ESN'
    elif corridor == -1 and direction == 1:
        category = 'ENS'
    elif corridor == 0 and direction == -1:
        category = 'CSN'
    elif corridor == 0 and direction == 1:
        category = 'CNS'
    elif corridor == 1 and direction == -1:
        category = 'OSN'
    elif corridor == 1 and direction == 1:
        category = 'ONS'
    else:
        print(f"Corridor = {corridor}, Direction = {direction}")
        continue

    pir0 = pir0.convert('L')
    pir1 = pir1.convert('L')
    pir2 = pir2.convert('L')
    pir3 = pir3.convert('L')
    pirs = [pir0, pir1, pir2, pir3]

    fig, ((p0, p2), (s0, s2), (p1, p3), (s1, s3)) = plt.subplots(4, 2, gridspec_kw={
        'height_ratios': [2, .3, 2, .3],
        'hspace': 0,
        'wspace': -0.1,
    },
                                                                 sharex='all')
    x_vals = [pir0.size[0] * (x / len(raw['pir0'])) for x in range(len(raw['pir0']))]
    fig.suptitle(f"Category: {category}" + '\n' + f"Speed: {round(speed * 3.6, 2)} km/h")

    p0.set_title(f"PIR 0")
    p0.imshow(pirs[0])
    p0.axis('off')

    s0.plot(x_vals, raw['pir0'])
    s0.axis('off')
    s0.set_xlim((-300, 1000))

    p1.set_title(f"PIR 1")
    p1.imshow(pirs[1])
    p1.axis('off')

    s1.plot(x_vals, raw['pir1'])
    s1.axis('off')

    p2.set_title(f"PIR 2")
    p2.imshow(pirs[2])
    p2.axis('off')

    s2.plot(x_vals, raw['pir2'])
    s2.axis('off')

    p3.set_title(f"PIR 3")
    p3.imshow(pirs[3])
    p3.axis('off')

    s3.plot(x_vals, raw['pir3'])
    s3.axis('off')

    # fig.tight_layout()
    plt.show()

    # plt.savefig(f"pir_images/{category}_{str(format(speed, '.6f')).replace('.', '-').replace(',', '-')}.png")
    # plt.clf()

    if mean_data[category]['pir0'] is None:
        mean_data[category]['pir0'] = np.array(pir0, dtype=np.uint64)
        mean_data[category]['pir1'] = np.array(pir1, dtype=np.uint64)
        mean_data[category]['pir2'] = np.array(pir2, dtype=np.uint64)
        mean_data[category]['pir3'] = np.array(pir3, dtype=np.uint64)
    else:
        mean_data[category]['pir0'] += np.asarray(pir0)
        mean_data[category]['pir1'] += np.asarray(pir1)
        mean_data[category]['pir2'] += np.asarray(pir2)
        mean_data[category]['pir3'] += np.asarray(pir3)

    # if category == 'OSN':
    #     print(mean_data[category]['pir0'])
    mean_data[category]['count'] += 1

for category in mean_data:
    try:
        mean_data[category]['pir0'] = mean_data[category]['pir0'] / mean_data[category]['count']
        mean_data[category]['pir1'] = mean_data[category]['pir1'] / mean_data[category]['count']
        mean_data[category]['pir2'] = mean_data[category]['pir2'] / mean_data[category]['count']
        mean_data[category]['pir3'] = mean_data[category]['pir3'] / mean_data[category]['count']
    except Exception as e:
        print("Error on", category)


for category in mean_data:
    try:
        plt.suptitle(f"Category: {category}")
        for pir in range(4):
            plt.subplot(2, 2, pir+1)
            plt.title(f"PIR {pir}")
            plt.imshow(mean_data[category][f"pir{pir}"])
            plt.axis('off')
        plt.savefig('pir_images/' + category + '.png', )
        # plt.show()
    except:
        print(category)
#
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

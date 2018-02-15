import numpy as np
import os

from PIL import Image

DATA_DIR = 'G:/DL/large_scale_geovisual_search/data'
data_folders_list = os.listdir(DATA_DIR)

print(len(data_folders_list))

for data_folder_name in data_folders_list:
	image_name = os.listdir(DATA_DIR + os.sep + data_folder_name)[0]
	text_name = os.listdir(DATA_DIR + os.sep + data_folder_name)[1]

	image = np.array(Image.open(DATA_DIR + os.sep + data_folder_name + os.sep + image_name))
	print(image.shape)

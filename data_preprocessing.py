import numpy as np
import cv2
import os

from PIL import Image

class GeotiffImageManipulator:
	def __init__(self, DATA_DIR):
		self.DATA_DIR = DATA_DIR
		self.data_folders_list = os.listdir(self.DATA_DIR)

	def divide(self, length_offset=500):
		for data_folder_name in self.data_folders_list:
			image_name = os.listdir(self.DATA_DIR + os.sep + data_folder_name)[0]
			text_name = os.listdir(self.DATA_DIR + os.sep + data_folder_name)[1]

			image = np.array(Image.open(self.DATA_DIR + os.sep + data_folder_name + os.sep + image_name))
			max_dim_x = image.shape[0]
			max_dim_y = image.shape[1]
			print(max_dim_x, max_dim_y)

			min_x, min_y, max_x, max_y = 0, 0, 0, 0
			while(max_y + length_offset <= max_dim_y):
				min_y = max_y
				max_y += length_offset

				while(max_x + length_offset <= max_dim_x):
					min_x = max_x
					max_x += length_offset

					print(min_x, min_y, max_x, max_y)
					new_file_name = image_name[0:-4] + '_' + str(min_x) + '_' + str(min_y) + '_' + str(max_x) + '_' + str(max_y) + '.jpg'
					cv2.imshow(self.DATA_DIR + os.sep + data_folder_name + os.sep + new_file_name, image[min_x:max_x, min_y:max_y, 1:4])
					cv2.waitKey(0)
					break
					min_x -= int(length_offset/2)
					max_x -= int(length_offset/2)

				min_y -= int(length_offset/2)
				max_y -= int(length_offset/2)
				min_x = 0
				max_x = 0



DATA_DIR = 'G:/DL/large_scale_geovisual_search/data'
geotiff_image_manipulator_instance = GeotiffImageManipulator(DATA_DIR)
geotiff_image_manipulator_instance.divide(length_offset=500)

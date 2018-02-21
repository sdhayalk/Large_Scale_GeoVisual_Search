import numpy as np
import cv2
import os
import h5py

from PIL import Image

class GeotiffImageManipulator:
	def __init__(self, DATA_DIR):
		'''Constructor to initialize root data directory
		Arguments:
			DATA_DIR {str} -- root data directory which contains 
		'''
		self.DATA_DIR = DATA_DIR
		self.data_folders_list = os.listdir(self.DATA_DIR)

	def divide(self, length_offset=128):
		'''funtion that slides a window of dimension length_offset x length_offset with a stride of length_offset/2
		   and saves the resultant image in the window in the corresponding folder.
		Keyword Arguments:
			length_offset {number} -- dimension of the divided image (same across length and height) (default: {128})
		'''
		for data_folder_name in self.data_folders_list:
			print('In folder', data_folder_name)
			image_name = os.listdir(self.DATA_DIR + os.sep + data_folder_name)[0]
			text_name = os.listdir(self.DATA_DIR + os.sep + data_folder_name)[1]

			image = np.array(Image.open(self.DATA_DIR + os.sep + data_folder_name + os.sep + image_name))
			max_dim_x = image.shape[0]
			max_dim_y = image.shape[1]

			min_x, min_y, max_x, max_y = 0, 0, 0, 0
			while(max_y + length_offset <= max_dim_y):
				min_y = max_y
				max_y += length_offset

				while(max_x + length_offset <= max_dim_x):
					min_x = max_x
					max_x += length_offset

					# print(min_x, min_y, max_x, max_y)
					new_file_name = image_name[0:-4] + '_' + str(min_x) + '_' + str(min_y) + '_' + str(max_x) + '_' + str(max_y) + '.jpg'
					Image.fromarray(image[min_x:max_x, min_y:max_y, 0:4]).save(self.DATA_DIR + os.sep + data_folder_name + os.sep + new_file_name)
					min_x -= int(length_offset/2)
					max_x -= int(length_offset/2)

				min_y -= int(length_offset/2)
				max_y -= int(length_offset/2)
				min_x = 0
				max_x = 0

	def delete_images(self):
		'''This function deletes all .jpg images in folders of root data directory.
		   This function can be useful if you want to recalculate and redivide images
		'''
		for data_folder_name in self.data_folders_list:
			files = os.listdir(self.DATA_DIR + os.sep + data_folder_name)
			print('deleting .jpg files in folder', data_folder_name)
			for file in files:
				if '.jpg' in file:
					os.remove(self.DATA_DIR + os.sep + data_folder_name + os.sep + file)


def convert_to_HDF5(hdf5_train_filename, data_dir, length_offset):
	dataset_train_features = []

	for data_folder_name in os.listdir(data_dir):
		print('In folder', data_folder_name)
		for file_name in os.listdir(data_dir + os.sep + data_folder_name):
			if '.jpg' in file_name:
				print(file_name)
				image_name = data_dir + os.sep + data_folder_name + os.sep + file_name
				image = np.array(Image.open(image_name))
				dataset_train_features.append(image)

	dataset_train_features = np.array(dataset_train_features).reshape((len(dataset_train_features), 3, length_offset, length_offset))
	
	with h5py.File(hdf5_train_filename, 'w') as f:
		f['data'] = dataset_train_features
		f['label'] = dataset_train_features


def main():
	DATA_DIR = 'G:/DL/large_scale_geovisual_search/data'
	length_offset = 128

	geotiff_image_manipulator_instance = GeotiffImageManipulator(DATA_DIR)
	# geotiff_image_manipulator_instance.delete_images()
	geotiff_image_manipulator_instance.divide(length_offset=length_offset)

	DATASET_TRAIN_HDF5_PATH = DATA_DIR + os.sep + 'dataset_train.hdf5'
	# convert_to_HDF5(DATASET_TRAIN_HDF5_PATH, DATA_DIR, length_offset)

if __name__ == '__main__':
	main()

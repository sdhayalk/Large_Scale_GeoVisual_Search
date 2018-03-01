import cv2
import numpy as np
import os
import h5py
import copy

from PIL import Image
from random import shuffle

class DataAugmentation:
	def __init__(self, RESIZE_DIM):
		self.RESIZE_DIM = RESIZE_DIM
		

	def flip_90(self, image):
		new_image = np.rot90(image)
		new_image = new_image.reshape((3, self.RESIZE_DIM, self.RESIZE_DIM))
		return new_image


	def flip_180(self, image):
		new_image = np.rot90(image)
		new_image = np.rot90(new_image)
		new_image = new_image.reshape((3, self.RESIZE_DIM, self.RESIZE_DIM))
		return new_image


	def flip_270(self, image):
		new_image = np.rot90(image)
		new_image = np.rot90(new_image)
		new_image = np.rot90(new_image)
		new_image = new_image.reshape((3, self.RESIZE_DIM, self.RESIZE_DIM))
		return new_image


	def flip_vertically(self, image):
		new_image = cv2.flip(image, 1)
		new_image = new_image.reshape((3, self.RESIZE_DIM, self.RESIZE_DIM))
		return new_image


	def flip_horizontally(self, image):
		new_image = cv2.flip(image, 0)
		new_image = new_image.reshape((3, self.RESIZE_DIM, self.RESIZE_DIM))
		return new_image


	def random_rotate(self, image):
		pass


	def random_zoom(self, image):
		pass


	def random_lightning(self, image):
		pass


class DataPreprocessing(DataAugmentation):

	def __init__(self, DATA_DIR, RESIZE_DIM):
		DataAugmentation.__init__(self, RESIZE_DIM)
		self.DATA_DIR = DATA_DIR


	def augment_and_convert_to_hdf5(self, train_validation_counter, hdf5_train_filename, hdf5_validation_filename, augmentation_factor=1, flip_90=True, flip_180=True, flip_270=True, flip_vertically=True, flip_horizontally=True, random_rotate=True, random_zoom=True, random_lightning=True):
		file_folder_list = []
		f_train = []
		f_test = []
		f_train_data = []
		f_train_label = []
		f_test_data = []
		f_test_label = []
		first_flag = []
		counter = 0
		round_robin_index = 0

		for folder_name_index in range(0, len(os.listdir(self.DATA_DIR))):
			folder_name = os.listdir(self.DATA_DIR)[folder_name_index]
			
			f_train.append( h5py.File(hdf5_train_filename[:-5] + str(folder_name_index) + hdf5_train_filename[-5:], 'w') )
			f_test.append( h5py.File(hdf5_validation_filename[:-5] + str(folder_name_index) + hdf5_validation_filename[-5:], 'w') )
			first_flag.append(True)
			f_train_data.append(None)
			f_train_label.append(None)
			f_test_data.append(None)
			f_test_label.append(None)

			for file_name in os.listdir(self.DATA_DIR + os.sep + folder_name):
				file_folder_list.append([folder_name_index, self.DATA_DIR + os.sep + folder_name + os.sep + file_name])

		shuffle(file_folder_list)

		for arr in file_folder_list:
			folder_name_index = arr[0]
			file_path = arr[1]

			print(file_path)
			image = None
			images_train_dataset_temp = []
			labels_train_dataset_temp = []

			if '.jpg' in file_path:
				image = cv2.imread(file_path)
			elif '.tif' in file_path:
				image = np.array(Image.open(file_path))

			image = cv2.resize(image, (self.RESIZE_DIM, self.RESIZE_DIM))
			new_image = image.copy()
			new_image = new_image.reshape((3, self.RESIZE_DIM, self.RESIZE_DIM))
			label = folder_name_index
			images_train_dataset_temp.append(new_image)
			labels_train_dataset_temp.append(label)

			# if flip_90:
			# 	new_image = self.flip_90(image.copy())
			# 	images_train_dataset_temp.append(new_image)
			# 	labels_train_dataset_temp.append(label)

			# if flip_180:
			# 	new_image = self.flip_180(image.copy())
			# 	images_train_dataset_temp.append(new_image)
			# 	labels_train_dataset_temp.append(label)

			# if flip_270:
			# 	new_image = self.flip_270(image.copy())
			# 	images_train_dataset_temp.append(new_image)
			# 	labels_train_dataset_temp.append(label)

			# if flip_vertically:
			# 	new_image = self.flip_vertically(image.copy())
			# 	images_train_dataset_temp.append(new_image)
			# 	labels_train_dataset_temp.append(label)

			# if flip_horizontally:
			# 	new_image = self.flip_horizontally(image.copy())
			# 	images_train_dataset_temp.append(new_image)
			# 	labels_train_dataset_temp.append(label)
			
			images_train_dataset_temp = np.array(images_train_dataset_temp, dtype='float')
			labels_train_dataset_temp = np.array(labels_train_dataset_temp, dtype='int')
			images_train_dataset_temp = images_train_dataset_temp / 255.0	

			num_images = 1 #* len(os.listdir(self.DATA_DIR + os.sep + folder_name))

			if first_flag[round_robin_index]:
				f_train_data[round_robin_index] = f_train[round_robin_index].create_dataset("data", (num_images,3,224,224), maxshape=(None,3,224,224), chunks=(num_images,3,224,224))
				f_train_label[round_robin_index] = f_train[round_robin_index].create_dataset("label", (num_images,), maxshape=(None,), chunks=(num_images,))
				f_test_data[round_robin_index] = f_test[round_robin_index].create_dataset("data", (num_images,3,224,224), maxshape=(None,3,224,224), chunks=(num_images,3,224,224))
				f_test_label[round_robin_index] = f_test[round_robin_index].create_dataset("label", (num_images,), maxshape=(None,), chunks=(num_images,))

				try:
					f_train_data[round_robin_index][:] = images_train_dataset_temp
					f_train_label[round_robin_index][:] = labels_train_dataset_temp
					f_test_data[round_robin_index][:] = images_train_dataset_temp
					f_test_label[round_robin_index][:] = labels_train_dataset_temp
				except:
					pass
				
			else:
				if counter != train_validation_counter:
					f_train_data[round_robin_index].resize(f_train_data[round_robin_index].shape[0] + num_images, axis=0)
					f_train_label[round_robin_index].resize(f_train_label[round_robin_index].shape[0] + num_images, axis=0)
					f_train_data[round_robin_index][-num_images:] = images_train_dataset_temp
					f_train_label[round_robin_index][-num_images:] = labels_train_dataset_temp
					counter += 1
				
				else:
					f_test_data[round_robin_index].resize(f_test_data[round_robin_index].shape[0] + num_images, axis=0)
					f_test_label[round_robin_index].resize(f_test_label[round_robin_index].shape[0] + num_images, axis=0)
					f_test_data[round_robin_index][-num_images:] = images_train_dataset_temp
					f_test_label[round_robin_index][-num_images:] = labels_train_dataset_temp
					counter = 0

			first_flag[round_robin_index] = False
			print('Saved image', file_name, '~ current trainhdf5 dshape:', f_train_data[round_robin_index].shape)
			round_robin_index += 1
			if round_robin_index >= len(os.listdir(self.DATA_DIR)):
				round_robin_index = 0




def main():
	# DATA_DIR = 'G:/DL/satellite_imagery_land_classification/data/mydataset'
	DATA_DIR = 'G:/DL/satellite_imagery_land_classification/data/UCMerced_LandUse/Images'
	RESIZE_DIM = 224
	hdf5_train_filename = 'G:/DL/satellite_imagery_land_classification/data/ucmerced_dataset_train.hdf5'
	hdf5_validation_filename = 'G:/DL/satellite_imagery_land_classification/data/ucmerced_dataset_validation.hdf5'
	train_validation_counter = 15


	data_preprocessing_instance = DataPreprocessing(DATA_DIR, RESIZE_DIM)
	data_preprocessing_instance.augment_and_convert_to_hdf5(train_validation_counter, hdf5_train_filename, hdf5_validation_filename)

if __name__ == '__main__':
	main()

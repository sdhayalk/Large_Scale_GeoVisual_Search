import caffe
import cv2
import numpy as np
import os

from PIL import Image

class Model:

	def __init__(self, CNN_NETWORK_PATH, CAFFEMODEL_PATH, DATA_DIR, USE_GPU=False):
		self.CNN_NETWORK_PATH = CNN_NETWORK_PATH
		self.CAFFEMODEL_PATH = CAFFEMODEL_PATH
		self.DATA_DIR = DATA_DIR
		self.data_folders_list = os.listdir(self.DATA_DIR)

		if USE_GPU:
			caffe.set_device(0)
			caffe.set_mode_gpu()
		else:
			caffe.set_mode_cpu()

		self.net = caffe.Net(self.CNN_NETWORK_PATH, self.CAFFEMODEL_PATH, caffe.TEST)


	def read_image_from_path(self, image_path):
		image = np.array(Image.open(image_path))
		image = cv2.resize(image, (224, 224))
		# image = image - 113.0
		# image = image / 255.0
		return image.reshape(1, 3, 224, 224)


	def test(self):
		for data_folder_name in self.data_folders_list[0:1]:
			files = os.listdir(self.DATA_DIR + os.sep + data_folder_name)
			print('forward pass of .jpg files in folder', data_folder_name)

			for file in files[0:3]:
				print(file)
				if '.jpg' in file:
					image = self.read_image_from_path(self.DATA_DIR + os.sep + data_folder_name + os.sep + file)
					output = self.net.forward(data=image)
					output = np.array(output['prob'])

					# for arr in output[0]:
						# print(arr[0][0])
					# print(output.shape)
					


def main():
	DATA_DIR = 'G:/DL/large_scale_geovisual_search/data'
	CNN_NETWORK_PATH = "ResNet-101-deploy.prototxt"		# for visualization, go to http://ethereon.github.io/netscope/#/gist/b21e2aae116dc1ac7b50
	CAFFEMODEL_PATH =  "G:/DL/large_scale_geovisual_search/models/ResNet-101-model.caffemodel"

	model = Model(CNN_NETWORK_PATH, CAFFEMODEL_PATH, DATA_DIR, USE_GPU=True)
	model.test()


if __name__ == '__main__':
	main()
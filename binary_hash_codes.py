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
		image = image - 112.0
		image = image / 255.0
		return image.reshape(3, 224, 224)


	def test(self, batch_size):
		with open('binary_codes.txt','w') as f:
			for data_folder_name in self.data_folders_list[1:]:
				files = os.listdir(self.DATA_DIR + os.sep + data_folder_name)
				print('forward pass of .jpg files in folder', data_folder_name)

				batch_counter = 0
				batch_image = []
				batch_file = []

				for file in files[0:12]:
					if '.jpg' in file:
						image = self.read_image_from_path(self.DATA_DIR + os.sep + data_folder_name + os.sep + file)
						batch_image.append(image)
						batch_file.append(file)
						batch_counter += 1

						if batch_counter == batch_size:
							batch_image = np.array(batch_image)
							output = self.net.forward(data=batch_image)
							output = np.array(output['pool5'])
							print(output)
							# for arr in output:
								# print(arr)
							# print(output.shape)

							for i in range(0, len(output)):
								code = ""
								for arr in output[i]:
									if arr[0][0] >= 0.5:
										code = code + '1'
									else:
										code = code + '0'

								f.write(code + ',' + batch_file[i])
								f.write('\n')

							batch_counter = 0
							batch_image = []
							batch_file = []

			f.close()


def main():
	DATA_DIR = 'G:/DL/large_scale_geovisual_search/data'
	CNN_NETWORK_PATH = "ResNet-101-deploy.prototxt"		# for visualization, go to http://ethereon.github.io/netscope/#/gist/b21e2aae116dc1ac7b50
	CAFFEMODEL_PATH =  "G:/DL/large_scale_geovisual_search/models/snapshot_iter_90.caffemodel"

	model = Model(CNN_NETWORK_PATH, CAFFEMODEL_PATH, DATA_DIR, USE_GPU=True)
	model.test(8)


if __name__ == '__main__':
	main()
import caffe
import cv2
import numpy as np

from PIL import Image

class Model:

	def __init__(self, CNN_NETWORK_PATH, CAFFEMODEL_PATH, USE_GPU=False):
		self.CNN_NETWORK_PATH = CNN_NETWORK_PATH
		self.CAFFEMODEL_PATH = CAFFEMODEL_PATH

		if USE_GPU:
			caffe.set_device(0)
			caffe.set_mode_gpu()
		else:
			caffe.set_mode_cpu()

		self.net = caffe.Net(self.CNN_NETWORK_PATH, self.CAFFEMODEL_PATH, caffe.TEST)



def main():

	CNN_NETWORK_PATH = "ResNet-101-deploy.prototxt"		# for visualization, go to http://ethereon.github.io/netscope/#/gist/b21e2aae116dc1ac7b50
	CAFFEMODEL_PATH =  "G:/DL/large_scale_geovisual_search/models/ResNet-101-model.caffemodel"

	model = Model(CNN_NETWORK_PATH, CAFFEMODEL_PATH, USE_GPU=False)
	model.test('G:/DL/large_scale_geovisual_search/data/m_3311702_se_11_1_20140515/m_3311702_se_11_1_20140515_0_0_128_128.jpg')



if __name__ == '__main__':
	main()
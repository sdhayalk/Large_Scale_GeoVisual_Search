import caffe
import cv2
import numpy as np
import os

from PIL import Image

class Model:

	def __init__(self, CNN_NETWORK_PATH, CNN_SOLVER_PATH, CAFFEMODEL_PATH, DATA_DIR, USE_GPU=False):
		self.CNN_NETWORK_PATH = CNN_NETWORK_PATH
		self.CNN_SOLVER_PATH = CNN_SOLVER_PATH
		self.CAFFEMODEL_PATH = CAFFEMODEL_PATH
		self.DATA_DIR = DATA_DIR
		self.data_folders_list = os.listdir(self.DATA_DIR)

		if USE_GPU:
			caffe.set_device(0)
			caffe.set_mode_gpu()
		else:
			caffe.set_mode_cpu()

		self.net = caffe.Net(self.CNN_NETWORK_PATH, self.CAFFEMODEL_PATH, caffe.TRAIN)

	
	def display_stats(self):
		print("Network layers information:")
		for name, layer in zip(self.net._layer_names, self.net.layers):
		    print("{:<7}: {:17s}({} blobs)".format(name, layer.type, len(layer.blobs)))
		print("Network blobs information:")
		for name, blob in self.net.blobs.items():
		    print("{:<7}: {}".format(name, blob.data.shape))
		print('self.net.inputs:', self.net.inputs)
		print('self.net.outputs:', self.net.outputs)
	

	def train(self):
		solver = caffe.get_solver(self.CNN_SOLVER_PATH)
		self.display_stats(self.net)
		solver.solve()


def main():
	DATA_DIR = 'G:/DL/large_scale_geovisual_search/data'
	CNN_NETWORK_PATH = 'ResNet-101-deploy.prototxt'		# for visualization, go to http://ethereon.github.io/netscope/#/gist/b21e2aae116dc1ac7b50
	CNN_SOLVER_PATH = ''
	CAFFEMODEL_PATH =  'G:/DL/large_scale_geovisual_search/models/ResNet-101-model.caffemodel'

	model = Model(CNN_NETWORK_PATH, CAFFEMODEL_PATH, DATA_DIR, USE_GPU=True)
	model.train()


if __name__ == '__main__':
	main()
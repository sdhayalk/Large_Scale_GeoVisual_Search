import caffe
import os


class Training:
	def __init__(self, CNN_NETWORK_PATH, CNN_SOLVER_PATH, CAFFEMODEL_PATH, SOLVERSTATE_PATH, USE_GPU=True, resume_training=False):
		self.CNN_NETWORK_PATH = CNN_NETWORK_PATH
		self.CNN_SOLVER_PATH = CNN_SOLVER_PATH
		self.CAFFEMODEL_PATH = CAFFEMODEL_PATH
		self.SOLVERSTATE_PATH = SOLVERSTATE_PATH
		self.resume_training = resume_training

		if USE_GPU:
		    caffe.set_device(0)
		    caffe.set_mode_gpu()
		else:
		    caffe.set_mode_cpu()

		if self.resume_training:
			self.net = caffe.Net(self.CNN_NETWORK_PATH, self.CAFFEMODEL_PATH, caffe.TRAIN)
		else:
			self.net = caffe.Net(self.CNN_NETWORK_PATH, caffe.TRAIN)
		self.solver = caffe.get_solver(self.CNN_SOLVER_PATH)


	def display_stats(self):
		print("Network layers information:")
		for name, layer in zip(self.net._layer_names, self.net.layers):
		    print("{:<7}: {:17s}({} blobs)".format(name, layer.type, len(layer.blobs)))
		print("Network blobs information:")
		for name, blob in self.net.blobs.items():
		    print("{:<7}: {}".format(name, blob.data.shape))
		print("self.net.inputs:", self.net.inputs)
		print("self.net.outputs:", self.net.outputs)


	def train(self):
		self.display_stats()
		if self.resume_training:
			self.solver.restore(self.SOLVERSTATE_PATH)
		self.solver.solve()


def main():
	CNN_NETWORK_PATH = "G:/DL/satellite_imagery_land_classification/Satellite_Imagery_Land_Classification/ResNet-train-val.prototxt"
	CNN_SOLVER_PATH = "G:/DL/satellite_imagery_land_classification/Satellite_Imagery_Land_Classification/ResNet-solver.prototxt"
	CAFFEMODEL_PATH =  "G:/DL/satellite_imagery_land_classification/data/snapshot_iter_6000.caffemodel"
	SOLVERSTATE_PATH =  "G:/DL/satellite_imagery_land_classification/data/snapshot_iter_6000.solverstate"
	USE_GPU = True
	resume_training = True

	training = Training(CNN_NETWORK_PATH, CNN_SOLVER_PATH, CAFFEMODEL_PATH, SOLVERSTATE_PATH, USE_GPU=USE_GPU, resume_training=resume_training)
	training.train()


if __name__ == "__main__":
	main()
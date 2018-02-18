import caffe


def main():
	USE_GPU = True
	if USE_GPU:
	    caffe.set_device(0)
	    caffe.set_mode_gpu()
	else:
	    caffe.set_mode_cpu()

	CNN_NETWORK_PATH = "ResNet-101-deploy.prototxt"
	CAFFEMODEL_PATH =  "G:/DL/large_scale_geovisual_search/models/ResNet-101-model.caffemodel"
	net = caffe.Net(CNN_NETWORK_PATH, CAFFEMODEL_PATH, caffe.TEST)


if __name__ == '__main__':
	main()
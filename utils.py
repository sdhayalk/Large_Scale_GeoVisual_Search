import csv


def search_test(BINARY_CODES_TXT_PATH, number):
	search_result_distance = []
	search_result_file_name = []
	query_image = None

	with open(BINARY_CODES_TXT_PATH) as file:
		binary_codes_data = list(csv.reader(file, delimiter=","))
		binary_code_search_query = int(binary_codes_data[number][0], 2)
		query_image = binary_codes_data[number][1]
		

		for line in binary_codes_data:
			binary_code = int(line[0], 2)
			xor = binary_code_search_query ^ binary_code

			xor_string = "{0:b}".format(xor)
			xor_distance = 0
			for char in xor_string:
				xor_distance += int(char)

			search_result_distance.append(xor_distance)
			search_result_file_name.append(line[1])

		file.close()


	search_result_distance, search_result_file_name = (list(x) for x in zip(*sorted(zip(search_result_distance, search_result_file_name), key=lambda pair: pair[0])))

	print("Query image: ", query_image)
	for i in range(0, len(search_result_distance)):
		print(search_result_distance[i], search_result_file_name[i])


BINARY_CODES_TXT_PATH = 'G:/DL/large_scale_geovisual_search/Large_Scale_GeoVisual_Search/binary_codes.txt'
number = 0
search_test(BINARY_CODES_TXT_PATH, number)

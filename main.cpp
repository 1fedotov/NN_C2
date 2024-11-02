#include <iostream>
#include <vector>
#include <chrono>
#include "mnist_loader.h"
#include "network.h"

#define TRAIN_IMAGES_PATH "mnist/train-images.idx3-ubyte"
#define TRAIN_LABELS_PATH "mnist/train-labels.idx1-ubyte"
#define TEST_IMAGES_PATH "mnist/t10k-images.idx3-ubyte"
#define TEST_LABELS_PATH "mnist/t10k-labels.idx1-ubyte"

int main() {
	mnist_loader loader;


	std::vector<std::pair<std::vector<float>, std::vector<float>>> train_data = loader.load(TRAIN_IMAGES_PATH, TRAIN_LABELS_PATH);
	std::vector<std::pair<std::vector<float>, std::vector<float>>> test_data = loader.load(TEST_IMAGES_PATH, TEST_LABELS_PATH);

	network Network(std::vector<int> {784, 30, 10});

	Network.populate(0.0, 1.0);

	Network.SGD(train_data, 30, 10, 3.0, &test_data);
	

	return 0;
}


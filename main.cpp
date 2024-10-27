#include <iostream>
#include <vector>
#include "mnist_loader.h"
#include "network.h"

#define TRAIN_IMAGES_PATH "mnist/train-images.idx3-ubyte"
#define TRAIN_LABELS_PATH "mnist/train-labels.idx1-ubyte"
#define TEST_IMAGES_PATH "mnist/t10k-images.idx3-ubyte"
#define TEST_LABELS_PATH "mnist/t10k-labels.idx1-ubyte"

int main() {
	std::cout << "hello, world!\n";

	mnist_loader loader;

	//loader.load(TRAIN_IMAGES_PATH, TRAIN_LABELS_PATH, TEST_IMAGES_PATH, TEST_LABELS_PATH);

	network Network(std::vector<int> {784, 30, 9});

	Network.print_weights(0);
	Network.print_biases(0);
	Network.print_weights(1);
	Network.print_biases(1);

	Network.populate(-3.0, 3.0);

	Network.print_weights(0);
	Network.print_biases(0);
	Network.print_weights(1);
	Network.print_biases(1);


	return 0;
}


#include "mnist_loader.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>

int swap_bytes(int i)
{
	void* addr = &i; // get an adress of the first byte of the integer
	char* buff = static_cast<char*>(addr); // treat the integer as a sequence of bytes
	int result = 0;

	// swap bytes 0 1 2 3 -> 3 2 1 0
	char c = buff[0];
	buff[0] = buff[3];
	buff[3] = c;
	c = buff[1];
	buff[1] = buff[2];
	buff[2] = c;

	return i;
}

//implementation of the MNIST dataset loader

void mnist_loader::load(const std::string& train_images_path, const std::string& train_labels_path, 
	const std::string& test_images_path, const std::string& test_labels_path)
{
	std::vector<std::pair<int, std::vector<uint8_t>>> train_data;

	std::ifstream train_images(train_images_path, std::ios::binary);
	std::ifstream train_labels(train_labels_path, std::ios::binary);

	if (train_labels.is_open() && train_images.is_open())
	{
		IMAGES_HEADER imagesHeader;
		LABELS_HEADER labelsHeader;

		// Reading headers
		train_images.read(reinterpret_cast<char*>(&imagesHeader), sizeof(imagesHeader));
		train_labels.read(reinterpret_cast<char*>(&labelsHeader), sizeof(labelsHeader));

		imagesHeader.magic_num = swap_bytes(imagesHeader.magic_num);
		imagesHeader.items_num = swap_bytes(imagesHeader.items_num);
		imagesHeader.row_num = swap_bytes(imagesHeader.row_num);
		imagesHeader.col_num = swap_bytes(imagesHeader.col_num);

		labelsHeader.magic_num = swap_bytes(labelsHeader.magic_num);
		labelsHeader.items_num = swap_bytes(labelsHeader.items_num);

		// Check headers format
		if (imagesHeader.magic_num != 2051)
		{
			std::cerr << "The images format is wrong\n";
		}
		if (labelsHeader.magic_num != 2049)
		{
			std::cerr << "The labels format is wrong\n";
		}
		if (imagesHeader.items_num != labelsHeader.items_num)
		{
			std::cerr << "Number of items doesn't math\n";
		}

		// Making an array of pairs {x, y} where 'x' is an image array and 'y' is a label
		// In other words, array of pairs of inputs and expected values for a neural network
		char c;
		int size = imagesHeader.row_num * imagesHeader.col_num;
		char* buff = new char[size];
		train_data.reserve(sizeof(size));

		std::cout << "row num: " << imagesHeader.row_num << " ";
		std::cout << "col num: " << imagesHeader.col_num << "\n";

		std::cout << "size of buffer: " << size << "\n";

		for (int i = 0; i < imagesHeader.items_num; i++)
		{
			train_labels.read(&c, sizeof(c));
			train_images.read(buff, sizeof(buff));

			std::vector<uint8_t> image;
			image.reserve(size);

			for (int j = 0; j < imagesHeader.row_num * imagesHeader.col_num; j++)
			{
				image.push_back((uint8_t)(buff + j));
			}

			train_data.push_back(std::make_pair((int)c, image));
			//std::cout << i << " pair is read\n";
		}

		delete[] buff;
	}
	else
	{
		std::cerr << "Failed to open train dataset";
	}

	train_images.close();
	train_labels.close();

	std::cout << "Number of pairs loaded:" << train_data.size() << "\n";

	std::vector<uint8_t> image = train_data[0].second;
	std::cout << "image pixels: " << image.size() << "\n";
	for (int i = 0; i < image.size(); i++)
	{
		if (i % 28 == 0 && i != 0) std::cout << "\n";
		std::cout << (int)image[i] << " ";
	}
}

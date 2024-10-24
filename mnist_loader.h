#pragma once
#include <string>

//files structures provided at http://yann.lecun.com/exdb/mnist/

struct IMAGES_HEADER
{
	int magic_num;
	int items_num;
	int row_num;
	int col_num;
};

struct LABELS_HEADER
{
	int magic_num;
	int items_num;
};

class mnist_loader
{
	public:
		void load(const std::string& train_images, const std::string& train_labels,
			const std::string& test_images, const std::string& test_labels);
};



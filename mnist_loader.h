#pragma once
#include <string>
#include <vector>
#include <Eigen/Dense>

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

struct DataSample
{
	Eigen::VectorXf label;
	Eigen::VectorXf image;
};

class mnist_loader
{
	public:
		std::vector<DataSample> load(const std::string& train_images, const std::string& train_labels);
};



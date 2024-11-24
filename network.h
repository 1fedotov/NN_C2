#pragma once
#include <vector>
#include "mnist_loader.h"

// pair<label, image> dataUnit
typedef std::pair<std::vector<float>, std::vector<float>> dataUnit;
typedef std::vector<std::vector<float>> weightMatrix;
typedef std::vector<float> biasesVec;

class network
{
	private:
		/*
		* vector<int> contains length of corresponding layer
		* weights tensors, where rows are current layer's neurons and cols are previous layer neurons
		*
		* biases are vector of vectors, where inner vector corresponds to i'th layers j'th neuron
		* array of vectors, vectors contain activations for neuron layer
		*/

		std::vector<int> layers;
		std::vector<Eigen::MatrixXf> weights;
		std::vector<Eigen::VectorXf> biases;

	public:
		// Initialize network with weights tensor and biases matrix, all values are 0's
		// First element is a size of the input vector, last element is output vector
		network(std::vector<int> layers_vec);

		// Populate network's weights and biases with random
		void populate(float min, float max);

		Eigen::VectorXf feedforward(const Eigen::VectorXf& input);

		void SGD(std::vector<DataSample>& train_data, int epochs, int mini_batch_size, float eta, const std::vector<DataSample>* test_data = nullptr);

		void update_mini_batch(const std::vector<DataSample>& mini_batch, float eta);

		std::pair<std::vector<Eigen::VectorXf>, std::vector<Eigen::MatrixXf>> backpropagate(const DataSample& train_data);

		int evaluate(const std::vector<DataSample>& test_data);

		// For debug purposes, shows the weights and biases for the 1 deep layer
		void log();

		void print_weights(int layer);

		void print_biases(int layer);
};


#pragma once

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
		std::vector<std::vector<std::vector<float>>> weights;
		std::vector<std::vector<float>> biases;
		std::vector<float>* a;

	public:
		// Initialize network with weights tensor and biases matrix, all values are 0's
		// First element is a size of the input vector, last element is output vector
		network(std::vector<int> layers_vec);

		// Populate network's weights and biases with random
		void populate();

		// For debug purposes, shows the weights and biases for the 1 deep layer
		void log();
};


#include <iostream>
#include <vector>
#include "network.h"
#include "algorithm"
#include "random"

network::network(std::vector<int> layers_vec)
{
	layers = layers_vec;

	// Initialize weights tensor for network
	// weights[i][j][k] is a weight from previuos layer k'th neuron 
	// incoming to j'th neuron in i'th layer
	for (int i = 1; i < layers.size(); i++)
	{
		std::vector<std::vector<float>> matrix;
		//matrix.reserve(layers[i] * sizeof(float));

		for (int j = 0; j < layers[i]; j++)
		{
			std::vector<float> row(layers[i - 1], 0);
			matrix.push_back(row);
		}

		weights.push_back(matrix);
	}

	// Initialize biases matrix b[i][j], bias of j'th neuron in i'th layer
	for (int i = 1; i < layers.size(); i++)
	{
		std::vector<float> column(layers[i], 0);
		biases.push_back(column);
	}

}

void network::populate()
{

}

void network::log()
{
	std::cout << "Number of layers: " << layers.size() << "\n";

	for (int i = 0; i < layers.size(); i++)
	{
		std::cout << i << " layer has " << layers[i] << " neurons\n";
	}

	std::cout << "Weights for 1 neuron layer: \n";
	std::cout << "{\n";
	for (int i = 0; i < weights[0].size(); i++)
	{
		std::cout << "[ ";
		for (int j = 0; j < weights[0][i].size(); j++)
		{
			std::cout << weights[0][i][j] << " ";
		}
		std::cout << "],\n";
	}
	std::cout << "}\n";

	std::cout << "Biases for 1 neuron layer: \n{\n";
	for (int i = 0; i < biases[0].size(); i++)
	{
		std::cout << biases[0][i] << "\n";
		
	}
	std::cout << "}\n";
}

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

void network::populate(float min, float max)
{
	std::default_random_engine eng;
	std::uniform_real_distribution<float> dist(min, max);

	// populate weights
	for (int i = 0; i < weights.size(); i++)
	{
		for (int j = 0; j < weights[i].size(); j++)
		{
			for (int k = 0; k < weights[i][j].size(); k++)
			{
				weights[i][j][k] = dist(eng);
			}
		}
	}

	// populate biases
	for (int i = 0; i < biases.size(); i++)
	{
		for (int j = 0; j < biases[i].size(); j++)
		{
			biases[i][j] = dist(eng);
		}
	}
}

void network::log()
{
	std::cout << "Number of layers: " << layers.size() << "\n";

	for (int i = 0; i < layers.size(); i++)
	{
		std::cout << i << " layer has " << layers[i] << " neurons\n";
	}
}

void network::print_weights(int layer)
{
	if (layer < 0 || layer >= layers.size() - 1)
	{
		std::cout << "index out of range" << "\n";
		return;
	}

	std::cout << "Weights for " << layer << " neuron layer : \n";
	std::cout << "{\n";
	for (int i = 0; i < weights[layer].size(); i++)
	{
		std::cout << "[ ";
		for (int j = 0; j < weights[layer][i].size(); j++)
		{
			std::cout << weights[layer][i][j] << " ";
		}
		std::cout << "],\n";
	}
	std::cout << "}\n";
}

void network::print_biases(int layer)
{
	if (layer < 0 || layer >= layers.size() - 1)
	{
		std::cout << "index out of range" << "\n";
		return;
	}

	std::cout << "Biases for " << layer << " neuron layer : \n{\n";
	for (int i = 0; i < biases[layer].size(); i++)
	{
		std::cout << biases[layer][i] << "\n";

	}
	std::cout << "}\n";
}

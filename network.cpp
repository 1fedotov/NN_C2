#include <iostream>
#include <vector>
#include "network.h"
#include "algorithm"
#include "random"
#include "util.h"

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

std::vector<float> network::feedforward(std::vector<float> input)
{
	std::vector<float> a = input;

	// Go through each layer in network, i'th layer
	for (int i = 0; i < biases.size(); i++)
	{
		std::vector<float> buff;

		// Go through each neuron in a layer, j'th neuron
		for (int j = 0; j < biases[i].size(); j++)
		{
			float dot_prod = 0;

			if (a.size() != weights[i][j].size())
			{
				throw "Feedforward error! Input size not equal weights";
			}

			// Go through each neuron's weight corresponding to the input 
			// from the previous layer neuron, k'th neuron
			for (int k = 0; k < weights[i][j].size(); k++)
			{
				dot_prod += a[k] * weights[i][j][k];
			}
			float z = dot_prod + biases[i][j];
			buff.push_back(sigmoid(z));
		}
		a = buff;
	}
	return a;
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

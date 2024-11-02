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
	std::random_device rd;
	std::default_random_engine eng(rd());
	std::normal_distribution<float> dist(0.0, 1.0);

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

std::vector<float> network::feedforward(const std::vector<float>& input)
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

			// Go through j'th neuron's weights corresponding to the input 
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

void network::SGD(std::vector<dataUnit>& train_data, int epochs, int mini_batch_size, float eta, const std::vector<dataUnit>* test_data)
{
	
	int n_test = test_data ? test_data->size() : 0;

	int n = train_data.size();
	for (int i = 0; i < epochs; i++)
	{
		std::default_random_engine eng;
		std::shuffle(train_data.begin(), train_data.end(), eng);

		std::vector<std::vector<dataUnit>> mini_batches;

		for (int j = 0; j < n / mini_batch_size; j++)
		{
			std::vector<dataUnit> mini_batch;
			for (int k = 0; k < mini_batch_size; k++)
			{
				mini_batch.push_back(train_data[k + mini_batch_size * j]);
			}
			mini_batches.push_back(mini_batch);
		}

		for (int j = 0; j < mini_batches.size(); j++)
		{
			update_mini_batch(mini_batches[j], eta);
		}

		if (test_data)
		{
			printf("Epoch %i: %i / %i\n", i, evaluate(*test_data), n_test);
		}
		else
		{
			printf("Epoch %i complete\n", i);
		}
	}
}

void network::update_mini_batch(const std::vector<dataUnit>& mini_batch, float eta)
{
	// create buffers for weights' and biases' nablas
	auto nabla_b = create_shape(biases);
	auto nabla_w = create_shape(weights);

	for (int i = 0; i < mini_batch.size(); i++)
	{
		auto delta_nabla = backpropagate(mini_batch[i]);
		add(nabla_b, delta_nabla.first);
		add(nabla_w, delta_nabla.second);
	}

	// update weights
	for (int i = 0; i < weights.size(); i++)
	{
		for (int j = 0; j < weights[i].size(); j++)
		{
			for (int k = 0; k < weights[i][j].size(); k++)
			{
				weights[i][j][k] = weights[i][j][k] - (eta / mini_batch.size()) * nabla_w[i][j][k];
			}
		}
	}

	// update biases
	for (int i = 0; i < biases.size(); i++)
	{
		for (int j = 0; j < biases[i].size(); j++)
		{
			biases[i][j] = biases[i][j] - (eta / mini_batch.size()) * nabla_b[i][j];
		}
	}

}

std::pair<std::vector<biasesVec>, std::vector<weightMatrix>> network::backpropagate(const dataUnit& train_data)
{
	// create buffers for weights' and biases' nablas
	auto nabla_b = create_shape(biases);
	auto nabla_w = create_shape(weights);

	std::vector<float> activation = train_data.second;
	std::vector<std::vector<float>> activations; // vector to store all activations vectors
	activations.push_back(activation);

	std::vector<std::vector<float>> zs; // vector to store all the weighted inputs vectors

	// feedforward
	for (int i = 0; i < biases.size(); i++)
	{
		std::vector<float> z;
		for (int j = 0; j < biases[i].size(); j++)
		{
			float wa = 0; // weight matrix and output vector dot product 
			for (int k = 0; k < weights[i][j].size(); k++)
			{
				wa += weights[i][j][k] * activation[k]; // calculates dot product of weights and activations
			}
			z.push_back(wa + biases[i][j]); // fills the vector with weighted input for a j'th neuron in i'th layer
		}
		zs.push_back(z);
		std::vector<float> current_activation;
		current_activation.reserve(z.size());

		for (int j = 0; j < z.size(); j++)
		{
			current_activation.push_back(sigmoid(z[j]));
		}
		activation = current_activation;
		activations.push_back(activation);
	}

	// backward pass
	std::vector<float> delta;
	delta.reserve(activations.back().size());
	for (int i = 0; i < activations.back().size(); i++)
	{
		delta.push_back(cost_deriv(activations.back()[i], train_data.first[i]) * sigmoid_deriv(zs.back()[i]));
	}
	
	for (int i = 0; i < nabla_b.back().size(); i++)
	{
		nabla_b.back()[i] = delta[i];
	}

	for (int i = 0; i < nabla_w.back().size(); i++)
	{
		for (int j = 0; j < nabla_w.back()[i].size(); j++)
		{
			nabla_w.back()[i][j] = delta[i] * activations[activations.size() - 2][j];
		}
	}

	for (int i = 2; i < layers.size(); i++)
	{
		std::vector<float> z = zs[zs.size() - i];
		std::vector<float> sp;
		sp.reserve(z.size());

		for (int j = 0; j < z.size(); j++)
		{
			sp.push_back(sigmoid_deriv(z[j]));
		}

		std::vector<float> new_delta;
		new_delta.reserve(sp.size());

		for (int j = 0; j < biases[biases.size() - i].size(); j++)
		{
			float wd = 0;
			for (int k = 0; k < delta.size(); k++)
			{
				wd += weights[weights.size() - i + 1][k][j] * delta[k];
			}
			new_delta.push_back(wd * sp[j]);
		}

		delta = new_delta;

		for (int j = 0; j < nabla_b[nabla_b.size() - i].size(); j++)
		{
			nabla_b[nabla_b.size() - i][j] = delta[j];
		}

		for (int j = 0; j < nabla_w[nabla_w.size() - i].size(); j++)
		{
			for (int k = 0; k < nabla_w[nabla_w.size() - i][j].size(); k++)
			{
				nabla_w[nabla_w.size() - i][j][k] = delta[j] * activations[activations.size() - i - 1][k];
			}
		}
	}

	return std::make_pair(nabla_b, nabla_w);

}

int network::evaluate(const std::vector<dataUnit>& test_data)
{
	int count = 0;
	for (int i = 0; i < test_data.size(); i++)
	{
		std::vector<float> result(feedforward(test_data[i].second));
		float rmax = 0;
		int idx = -1;
		int label = 0;
		for (int j = 0; j < result.size(); j++)
		{
			if (result[j] > rmax)
			{
				rmax = result[j];
				idx = j;
			}

			if (test_data[i].first[j] == 1) label = j;
		}
		if (idx == label) count++;
	}
	return count;
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

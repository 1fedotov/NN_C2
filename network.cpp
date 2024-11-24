#include <iostream>
#include <vector>
#include <chrono>
#include <Eigen/Dense>
#include "network.h"
#include "algorithm"
#include "random"
#include "util.h"

network::network(std::vector<int> layers_vec)
{
	layers = layers_vec;

	// Initialize weights matrices for network
	// weights[i][j][k] is a weight from previuos layer k'th neuron 
	// incoming to j'th neuron in i'th layer
	for (int i = 1; i < layers.size(); i++)
	{
		weights.push_back(Eigen::MatrixXd(layers[i], layers[i - 1]));
	}

	// Initialize biases vectors b[i][j], bias of j'th neuron in i'th layer
	for (int i = 1; i < layers.size(); i++)
	{
		biases.push_back(Eigen::VectorXd(layers[i]));
	}
}

void network::populate(float min, float max)
{
	std::random_device rd;
	std::default_random_engine eng(rd());
	std::normal_distribution<double> dist(0.0, 1.0);

	// populate weights
	for (int i = 0; i < weights.size(); i++)
	{
		for (auto& x : weights[i].reshaped()) x = dist(eng);
	}

	// populate biases
	for (int i = 0; i < biases.size(); i++)
	{
		for (auto& x : biases[i]) x = dist(eng);
	}
}

Eigen::VectorXd network::feedforward(const Eigen::VectorXd& input)
{
	Eigen::VectorXd a = input;

	// Go through each layer in network, i'th layer
	for (int i = 0; i < layers.size() - 1; i++)
	{
		a = (weights[i] * a + biases[i]).unaryExpr(&sigmoid);
	}
	return a;
}

void network::SGD(std::vector<DataSample>& train_data, int epochs, int mini_batch_size, double eta, const std::vector<DataSample>* test_data)
{
	
	int n_test = test_data ? test_data->size() : 0;

	int n = train_data.size();
	for (int i = 0; i < epochs; i++)
	{
		auto start = std::chrono::high_resolution_clock::now();

		std::default_random_engine eng;
		std::shuffle(train_data.begin(), train_data.end(), eng);

		std::vector<std::vector<DataSample>> mini_batches;
		mini_batches.reserve(n / mini_batch_size);

		for (int j = 0; j < n / mini_batch_size; j++)
		{
			mini_batches.push_back(slice(train_data, j * mini_batch_size, j * mini_batch_size + mini_batch_size));
		}

		for (int j = 0; j < mini_batches.size(); j++)
		{
			update_mini_batch(mini_batches[j], eta);
		}

		auto end = std::chrono::high_resolution_clock::now();

		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()/1000000.0;

		if (test_data)
		{
			printf("Epoch %i: %i / %i took %f sec\n", i, evaluate(*test_data), n_test, duration);
		}
		else
		{
			printf("Epoch %i complete took %f sec\n", i, duration);
		}
	}
}

void network::update_mini_batch(const std::vector<DataSample>& mini_batch, double eta)
{
	// create buffers for weights' and biases' nablas
	auto nabla_b = create_shape(biases);
	auto nabla_w = create_shape(weights);

	for (int i = 0; i < mini_batch.size(); i++)
	{
		auto delta_nabla = backpropagate(mini_batch[i]);
		for (int i = 0; i < nabla_b.size(); i++)
		{
			nabla_b[i] += delta_nabla.first[i];
			nabla_w[i] += delta_nabla.second[i];
		}
	}

	// update weights
	for (int i = 0; i < weights.size(); i++)
	{
		weights[i] -= (eta / mini_batch.size()) * nabla_w[i];
	}

	// update biases
	for (int i = 0; i < biases.size(); i++)
	{
		biases[i] -= (eta / mini_batch.size()) * nabla_b[i];
	}

}

std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::MatrixXd>> network::backpropagate(const DataSample& train_data)
{
	// create buffers for weights' and biases' nablas
	auto nabla_b = create_shape(biases);
	auto nabla_w = create_shape(weights);

	Eigen::VectorXd activation = train_data.image;
	std::vector<Eigen::VectorXd> activations; // vector to store all activations vectors
	activations.push_back(activation);

	std::vector<Eigen::VectorXd> zs; // vector to store all the weighted inputs vectors

	// feedforward
	for (int i = 0; i < layers.size() - 1; i++)
	{
		Eigen::VectorXd z;

		z = (weights[i] * activation + biases[i]).eval();

		zs.push_back(z);

		activation = z.unaryExpr(&sigmoid).eval();

		activations.push_back(activation);
	}

	// backward pass
	Eigen::VectorXd delta;

	delta = (cost_deriv(activations.back(), train_data.label).array() * zs.back().unaryExpr(&sigmoid_deriv).array()).eval();
	
	nabla_b.back() = delta;
	
	nabla_w.back() = (delta * activations[activations.size() - 2].transpose()).eval();
	
	// variable i in the loop corresponds to 
	// the i'th index from the end of the vectors
	for (int i = 2; i < layers.size(); i++)
	{
		Eigen::VectorXd z = zs[zs.size() - i];

		Eigen::VectorXd sp = z.unaryExpr(&sigmoid_deriv).eval();

		delta = ((weights[weights.size() - i + 1].transpose() * delta).eval().array() * sp.array()).eval();

		nabla_b[nabla_b.size() - i] = delta;
		
		nabla_w[nabla_w.size() - i] = (delta * activations[activations.size() - i - 1].transpose()).eval();
	}

	return std::make_pair(nabla_b, nabla_w);

}

int network::evaluate(const std::vector<DataSample>& test_data)
{
	int count = 0;
	for (int i = 0; i < test_data.size(); i++)
	{
		Eigen::VectorXd result(feedforward(test_data[i].image));

		int resMax;
		int testMax;

		result.maxCoeff(&resMax);
		test_data[i].label.maxCoeff(&testMax);

		if (resMax == testMax) count++;
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
	std::cout << weights[layer];
}

void network::print_biases(int layer)
{
	if (layer < 0 || layer >= layers.size() - 1)
	{
		std::cout << "index out of range" << "\n";
		return;
	}

	std::cout << "Biases for " << layer << " neuron layer : \n{\n";
	
	std::cout << biases[layer];
}

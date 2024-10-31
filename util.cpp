#include "util.h"
#include "network.h"
#include <math.h>
#include <vector>

float sigmoid(float z)
{
	return 1.0/(1.0 + exp(-z));
}

float sigmoid_deriv(float z)
{
	return exp(-z)/pow((1 + exp(-z)), 2);
}

float square_error(float y, float x)
{
	return 0.0f;
}

float cost_deriv(float a, float y)
{
	return (a - y);
}

float cross_entropy(float y, float x)
{
	return 0.0f;
}

std::vector<weightMatrix> create_shape(std::vector<weightMatrix> weights)
{
	for (int i = 0; i < weights.size(); i++)
	{
		for (int j = 0; j < weights[i].size(); j++)
		{
			for (int k = 0; k < weights[i][j].size(); k++)
			{
				weights[i][j][k] = 0;
			}
		}
	}

	return weights;
}

std::vector<biasesVec> create_shape(std::vector<biasesVec> biases)
{
	for (int i = 0; i < biases.size(); i++)
	{
		for (int j = 0; j < biases[i].size(); j++)
		{
			biases[i][j] = 0;
		}
	}

	return biases;
}

void add(std::vector<weightMatrix>& weights1, std::vector<weightMatrix>& weights2)
{
	for (int i = 0; i < weights1.size(); i++)
	{
		for (int j = 0; j < weights1[i].size(); j++)
		{
			for (int k = 0; k < weights1[i][j].size(); k++)
			{
				weights1[i][j][k] += weights2[i][j][k];
			}
		}
	}
}

void add(std::vector<biasesVec>& biases1, std::vector<biasesVec>& biases2)
{
	for (int i = 0; i < biases1.size(); i++)
	{
		for (int j = 0; j < biases1[i].size(); j++)
		{
			biases1[i][j] += biases2[i][j];
		}
	}
}

#pragma once
#include "network.h"
#include <vector>

float sigmoid(float z);

float sigmoid_deriv(float z);

float square_error(float y, float x);

float cost_deriv(float a, float y);

float cross_entropy(float y, float x);

std::vector<weightMatrix> create_shape(std::vector<weightMatrix> weights);

std::vector<biasesVec> create_shape(std::vector<biasesVec> biases);

void add(std::vector<weightMatrix>& weights1, std::vector<weightMatrix>& weights2);

void add(std::vector<biasesVec>& biases1, std::vector<biasesVec>& biases2);
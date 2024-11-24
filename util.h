#pragma once
#include "network.h"
#include <vector>

float sigmoid(float z);

float sigmoid_deriv(float z);

float square_error(float y, float x);

Eigen::VectorXf cost_deriv(Eigen::VectorXf a, Eigen::VectorXf y);

float cross_entropy(float y, float x);

template <typename T>
std::vector<T> slice(std::vector<T> const& v, int x, int y)
{
	auto begin = v.begin() + x;
	auto end = v.begin() + y;

	return std::vector<T>(begin, end);
}

template <typename T>
std::vector<T> create_shape(const std::vector<T>& elements) 
{
    std::vector<T> res(elements);

    for (auto& elem : res) 
    {
        for (auto& x : elem.reshaped()) x = 0;
    }

    return res;
}

template <typename T>
void add(std::vector<T>& result, std::vector<T> const& additive)
{
    if (result.size() != additive.size()) {
        throw std::invalid_argument("Vectors must have the same size.");
    }

    for (int i = 0; i < result.size(); i++)
    {
        result[i] += additive[i];
    }
}
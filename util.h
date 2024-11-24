#pragma once
#include "network.h"
#include <vector>

double sigmoid(double z);

double sigmoid_deriv(double z);

double square_error(double y, double x);

Eigen::VectorXd cost_deriv(Eigen::VectorXd a, Eigen::VectorXd y);

double cross_entropy(double y, double x);

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
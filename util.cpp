#include "util.h"
#include "network.h"
#include <math.h>
#include <vector>
#include <Eigen/Dense>

double sigmoid(double z)
{
	return 1.0/(1.0 + exp(-z));
}

double sigmoid_deriv(double z)
{
	return exp(-z)/pow((1 + exp(-z)), 2);
}

double square_error(double y, double x)
{
	return 0.0f;
}

Eigen::VectorXd cost_deriv(Eigen::VectorXd a, Eigen::VectorXd y)
{
	return a - y;
}

double cross_entropy(double y, double x)
{
	return 0.0f;
}


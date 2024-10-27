#include "util.h"
#include <math.h>

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

float cross_entropy(float y, float x)
{
	return 0.0f;
}
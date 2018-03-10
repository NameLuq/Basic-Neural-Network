#ifndef NETWORK_H
#define NETWORK_H
#include <armadillo>
#include <vector>

class Network {
public:

	std::vector<arma::Col<double>> neuron_layers;
	std::vector<arma::Mat<double>> vec_weights;
	std::vector<arma::Col<double>> vec_bias_weights;

	double lr = 3.0;

	Network(int, std::vector<int>, int);
	arma::Col<double> feedforward(arma::Col<double>);
	void train();
	void backpropagation();
	double func(double);
};

#endif
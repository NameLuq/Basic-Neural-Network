#ifndef NETWORK_H
#define NETWORK_H
#include <armadillo>
#include <vector>

class Network {
public:

	std::vector<arma::Col<double>> neuron_layers;
	std::vector<arma::Mat<double>> vec_weights;
	double lr = 1;

	Network(std::vector<int>);

	arma::Col<double> feedforward(arma::Col<double>);
	void train(arma::Col<double>, arma::Col<double>);

	static double sigmoid(double z) { return 1.0 / (1.0 + exp(-z)); }
};

#endif
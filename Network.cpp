#include "Network.h"
#include <iostream>

using namespace std;
using namespace arma;

double Network::func(double x) {
	return 1.0 / (1.0 + exp(-x));
}

Network::Network(int input, vector<int> neurons, int output) {

	for (int i = 0; i < neurons.size(); i++) {
		neuron_layers.push_back(vec(neurons[i]));
		vec_bias_weights.push_back(vec(neurons[i]));
	}

	vec_bias_weights.push_back(vec(output));

	vec_weights.push_back(mat(neurons[0], input));
	for (int i = 0; i < neurons.size() - 1; ++i) {
		vec_weights.push_back(mat(neurons[i + 1], neurons[i]));
	}
	vec_weights.push_back(mat(output, neurons[neurons.size() - 1]));

	//testing
	for (auto& a : vec_bias_weights) {
		a = a.randu();
		a.print();
		cout << "bias\n";
	}
	for (auto& a : neuron_layers) {
		a.print();
		cout << "vec\n";
	}
	cout << "fuck\n";
	for (auto& a : vec_weights) {
		a = a.randu() * 2 - 1.0;
		a.print();
		cout << "mat\n";
	}

}

Col<double> Network::feedforward(Col<double> input) {
	neuron_layers[0] = vec_bias_weights[0] + vec_weights[0] * input;
	/*for (int i = 0; i < vec_weights.size(); i++)
		;*/
	cout << "GUESS" << endl;
	neuron_layers[0].print();
}

void Network::train() {

}

void Network::backpropagation() {

}

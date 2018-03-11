#include "Network.h"
#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;
using namespace arma;

Network::Network(vector<int> neurons) {

	srand(time(0));

	int n = neurons.size();

	for (int i = 0; i < n - 1; i++) {
		neuron_layers.push_back(vec(neurons[i] + 1));
	}
	neuron_layers.push_back(vec(neurons[n - 1]));///output without bias

	for (int i = 0; i < n - 2; ++i) {
		vec_weights.push_back(mat(neurons[i] + 1, neurons[i + 1] + 1));
	}
	vec_weights.push_back(mat(neurons[n - 2] + 1, neurons[n - 1]));///last hidden layer-output without bias

	for (auto& a : neuron_layers)///setting the bias
		a(a.n_rows - 1) = 1;

	for (auto& a : vec_weights)
		a.for_each([](double & x) { x = -1 + rand() / ( RAND_MAX / 2.0 ); });///setting random values
}

Col<double> Network::feedforward(Col<double> input) {

	for (int i = 0; i < (int)input.size(); i++) {///adding input into network
		neuron_layers[0](i) = input(i);
	}

	int n = vec_weights.size() + 1;
	for (int i = 1; i < n - 1; i++) {
		neuron_layers[i] = vec_weights[i - 1].t() * neuron_layers[i - 1];////calculating weights
		neuron_layers[i].for_each([](double & x) { x = sigmoid(x); });////sigmoid
		neuron_layers[i](neuron_layers[i].n_rows - 1) = 1;////turning the bias to 1
	}
	neuron_layers[n - 1] = vec_weights[n - 2].t() * neuron_layers[n - 2];//calculating output
	neuron_layers[n - 1].for_each([](double & x) { x = sigmoid(x); });

	//cout << "GUESS ";

	//neuron_layers[n - 1].print();
	return neuron_layers[n - 1];
}

void Network::train(Col<double> input, Col<double> answer) {
	Col<double> output = this->feedforward(input);
	Col<double> error = answer - output;

	for (int i = vec_weights.size() - 1; i >= 0; i--) {
		Mat<double> delta = lr * error % neuron_layers[i + 1] % (1 - neuron_layers[i + 1]) * neuron_layers[i].t();
		error = vec_weights[i] * error;
		vec_weights[i] += delta.t();
	}
}


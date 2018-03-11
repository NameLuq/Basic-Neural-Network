#include <iostream>
#include <ctime>
#include <cstdlib>
#include "Network.h"

//#define ARMA_NO_DEBUG
#define ARMA_USE_WRAPPER

using namespace std;
using namespace arma;

int main(int argv, char** args) {

	srand(time(0));

	vector<int> layers{2, 4, 4, 1};
	Network net(layers);

	vector<Col<double>> inputs{{1, 1}, {1, 0}, {0, 1}, {0, 0}};
	vector<Col<double>> answers{{0}, {1}, {1}, {0}};

	for (int i = 0; i < 500; i++) {
		cout << "epoch " << i << endl;
		for (int k = 0; k < 4; k++)
			net.train(inputs[k], answers[k]);
		cout << endl;
	}

	cout << "final test: " << endl;
	net.feedforward(inputs[0]);//0

	return 0;
}
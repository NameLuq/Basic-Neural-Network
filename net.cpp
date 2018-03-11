#include "Network.h"

using namespace std;
using namespace arma;

int main(int argv, char** args) {

	vector<int> layers{2, 4, 4, 1};
	Network net(layers);

	vector<Col<double>> inputs{{1, 1}, {1, 0}, {0, 1}, {0, 0}};
	vector<Col<double>> answers{{0}, {1}, {1}, {0}};

	for (int i = 0; i < 1000; i++) {
		//cout << "stage " << i << endl;
		int k = rand() % 4;
		//cout << "ANSWER " << answers[k];
		net.train(inputs[k], answers[k]);
		//cout << endl;
	}

	for (int i = 0; i < 4; i++)
		net.feedforward(inputs[i]).print();

	return 0;
}
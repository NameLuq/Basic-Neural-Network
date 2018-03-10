#include <iostream>
#include <ctime>
#include <cstdlib>
#include "Network.h"

#define ARMA_NO_DEBUG

using namespace std;
using namespace arma;

int main(int argv, char** args) {

	srand(time(0));

	vector<int> layers{4, 4};
	Network net(2, layers, 2);

	Col<double> input{1, 0};
	net.feedforward(input);


	return 0;
}
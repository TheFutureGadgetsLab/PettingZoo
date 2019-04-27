#include <iostream>
#include <armadillo>
#include <gamelogic.hpp>
#include <NeuralNetwork.hpp>
#include <chrono> 
#include <fstream>
#include <random>

int main(int argc, const char **argv) {
    Params params;

    params.in_h = 3;
    params.in_w = 3;
    params.hlc = 2;
    params.npl = 5;

    NeuralNetwork parentA(params), parentB(params), childA(params), childB(params);

    parentA.seed(1);
    parentB.seed(2);
    childA.seed(3);
    childB.seed(4);

    parentA.generate();
    parentB.generate();

    parentA.print();
    parentB.print();


    childA.print();
    childB.print();
    
    return 0;
}
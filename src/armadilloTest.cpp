#include <iostream>
#include <armadillo>
#include <gamelogic.hpp>
#include <NeuralNetwork.hpp>
#include <chrono> 
#include <fstream>
#include <random>

int main(int argc, const char **argv) {
    Params params;

    params.in_h = 2;
    params.in_w = 2;
    params.hlc = 2;
    params.npl = 8;

    NeuralNetwork parentA(params), parentB(params), childA(params), childB(params);

    parentA.seed(1);
    parentB.seed(2);
    childA.seed(3);
    childB.seed(4);

    parentA.generate();
    parentB.generate();

    parentA.print();
    parentB.print();

    breed(parentA, parentB, childA, childB, 1);

    childA.print();
    childB.print();
    
    arma::Mat<double> A = arma::randu(3, 3);
    arma::Mat<double> B = arma::randu(3, 3);
    
    std::cout << A << std::endl;
    std::cout << B << std::endl;

    int splitLoc = 2;

    std::cout << arma::join_vert(A.rows(0, splitLoc - 1), B.rows(splitLoc, B.n_rows - 1)) << std::endl;
    std::cout << arma::join_vert(B.rows(0, splitLoc - 1), A.rows(splitLoc, A.n_rows - 1)) << std::endl;

    return 0;
}
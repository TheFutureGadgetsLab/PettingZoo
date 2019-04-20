#include <iostream>
#include <armadillo>
#include <gamelogic.hpp>
#include <chromosome.hpp>
#include <neural_network.hpp>
#include <NeuralNetwork.hpp>
#include <chrono> 

int main(int argc, const char **argv) {
    // Initialize the random generator
    // arma::arma_rng::set_seed_random();
    Params params;
    params.in_h = 2;
    params.in_w = 2;
    params.hlc  = 2;
    params.npl  = 4;

    NeuralNetwork pAn(params), pBn(params), cAn(params), cBn(params);
    Chromosome pAc(params), pBc(params), cAc(params), cBc(params);
    
    pAn.seed(1);
    pBn.seed(2);
    cAn.seed(3);
    cBn.seed(4);

    pAc.seed(1);
    pBc.seed(2);
    cAc.seed(3);
    cBc.seed(4);

    pAn.generate();
    pBn.generate();
    cAn.generate();
    cBn.generate();

    pAc.generate();
    pBc.generate();
    cAc.generate();
    cBc.generate();

    pAn.print();
    pBn.print();

    breed(pAn, pBn, cAn, cBn, 4);

    cAc.print();
    cAn.print();

    return 0;
}
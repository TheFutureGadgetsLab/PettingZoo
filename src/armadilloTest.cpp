#include <iostream>
#include <armadillo>
#include <gamelogic.hpp>
#include <NeuralNetwork.hpp>
#include <chrono> 
#include <fstream>
#include <random>

int main(int argc, const char **argv) {
    std::mt19937 engine;  // Mersenne twister random number engine
    std::uniform_real_distribution<float> chance(-1.0, 1.0);
    std::uniform_real_distribution<float> weight(0.0, 1.0);

    // arma::arma_rng::set_seed_random();
    arma::Mat<float> A = arma::randu<arma::Mat<float>>(5, 5);
    arma::Mat<float> mutateMatrix = arma::Mat<float>(5, 5);

    mutateMatrix.imbue( [&]() { if (chance(engine) < 0.5) return weight(engine); else return 1.0f;} );

    std::cout << A << std::endl;
    std::cout << mutateMatrix << std::endl;
    std::cout << A % mutateMatrix << std::endl;

    return 0;
}
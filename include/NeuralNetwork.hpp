#ifndef NN_HPP
#define NN_HPP

#include <defs.hpp>
#include <armadillo>
#include <gamelogic.hpp>
#include <vector>

class NeuralNetwork{
public:

    int inW, inH, hlc, npl;
    std::minstd_rand engine;

    arma::Mat<float> inputLayer;
    std::vector<arma::Mat<float>> hiddenLayers;
    arma::Mat<float> outputLayer;

    NeuralNetwork(Params params);

    void seed(unsigned int seed);
    void generate();
    void print();
    void evaluate(Game& game, Player& player);
};

#endif
#ifndef NN_HPP
#define NN_HPP

#include <defs.hpp>
#include <armadillo>
#include <gamelogic.hpp>
#include <vector>

class NeuralNetwork{
public:

    int inW, inH, hlc, npl;
    float fitness;
    int deathType;

    std::minstd_rand engine;

    arma::Mat<float> inputLayer;
    std::vector<arma::Mat<float>> hiddenLayers;
    arma::Mat<float> outputLayer;

    NeuralNetwork(Params params);
    NeuralNetwork(std::string fname);

    void seed(unsigned int seed);
    void generate();
    void print();
    void evaluate(Game& game, Player& player);
    void mutate(float mutateRate);
    void writeToFile(std::string fname, unsigned int level_seed);
};

void breed(NeuralNetwork& parentA, NeuralNetwork& parentB, NeuralNetwork& childA, NeuralNetwork& childB, unsigned int seed, int breedType);
unsigned int getStatsFromFile(std::string fname, Params& params);

#endif
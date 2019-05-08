#ifndef FFNN_HPP
#define FFNN_HPP

#include <defs.hpp>
#include <armadillo>
#include <gamelogic.hpp>
#include <vector>

class FFNN{
public:

    int inW, inH, hlc, npl;
    float fitness;
    int deathType;

    std::minstd_rand engine;

    arma::Mat<float> inputLayer;
    std::vector<arma::Mat<float>> hiddenLayers;
    arma::Mat<float> outputLayer;

    FFNN(Params params);
    FFNN(std::string fname);

    void seed(unsigned int seed);
    void generate();
    void print();
    void evaluate(Game& game, Player& player);
    void mutate(float mutateRate);
    void writeToFile(std::string fname, unsigned int level_seed);
};

void breed(FFNN& parentA, FFNN& parentB, FFNN& childA, FFNN& childB, unsigned int seed, int breedType);
unsigned int getStatsFromFile(std::string fname, Params& params);

#endif
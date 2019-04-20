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
    // arma::Mat<float> C = arma::Mat<float>(dim, dim);
    Params params;
    Game game;
    Player player;

    NeuralNetwork net(params);
    Chromosome chrom(params);
    
    net.seed(1);
    chrom.seed(1);

    net.generate();
    chrom.generate();
    
    game.genMap(1);

    evaluate_frame(game, player, chrom);

    net.evaluate(game, player);

    return 0;
}
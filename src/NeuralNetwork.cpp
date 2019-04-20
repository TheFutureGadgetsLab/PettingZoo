#include <armadillo>
#include <vector>
#include <random>
#include <defs.hpp>
#include <gamelogic.hpp>
#include <NeuralNetwork.hpp>

NeuralNetwork::NeuralNetwork(Params params)
{
    inW = params.in_w;
    inH = params.in_h;
    hlc = params.hlc;
    npl = params.npl;

    inputLayer = arma::Mat<float>(npl, inW * inH);

    hiddenLayers.resize(hlc - 1);
    for (arma::Mat<float>& layer : hiddenLayers) {
        layer = arma::Mat<float>(npl, npl);
    }

    outputLayer = arma::Mat<float>(BUTTON_COUNT, npl);
}

void NeuralNetwork::generate()
{
    std::uniform_real_distribution<> weightGen(-1.0, 1.0);

    for (int row = 0; row < inputLayer.n_rows; row++) {
        for (int col = 0; col < inputLayer.n_cols; col++) {
            inputLayer(row, col) =  weightGen(engine);
        }
    }

    for (arma::Mat<float>& layer : hiddenLayers) {
        for (int row = 0; row < layer.n_rows; row++) {
            for (int col = 0; col < layer.n_cols; col++) {
                layer(row, col) =  weightGen(engine);
            }
        }
    }

    for (int row = 0; row < outputLayer.n_rows; row++) {
        for (int col = 0; col < outputLayer.n_cols; col++) {
            outputLayer(row, col) =  weightGen(engine);
        }
    }
}

void NeuralNetwork::seed(unsigned int seed)
{
    engine.seed(seed);
}

void NeuralNetwork::print()
{
    printf("-------------------------------------------\n");
    int r, c, hl;

    printf("\nInput to first hidden layer adj:\n");
    std::cout << inputLayer << std::endl << std::endl;

    for (hl = 0; hl < this->hlc - 1; hl++) {
        printf("Hidden layer %d to %d act:\n", hl + 1, hl + 2);
        std::cout << hiddenLayers[hl] << std::endl << std::endl;
    }

    printf("Hidden layer %d to output act:\n", this->hlc);
    std::cout << outputLayer << std::endl << std::endl;


    printf("\nChromosome:\n");
    printf("in_w:\t%d\nin_h:\t%d\nnpl:\t%d\nhlc:\t%d\n", this->inH, this->inW, this->npl, this->hlc);
    printf("Size: %lld bytes\n", (inputLayer.size() + hiddenLayers.size() * hiddenLayers[0].size() + outputLayer.size()) * sizeof(float));
    printf("-------------------------------------------\n");
}

void NeuralNetwork::evaluate(Game& game, Player& player)
{
    arma::Mat<float> output, netInput;
    std::vector<float> tiles(inW * inH);

    game.getInputTiles(player, tiles, inH, inW);
    netInput = arma::Mat<float>(tiles);

    output = arma::tanh(inputLayer * netInput);

    for (arma::Mat<float>& layer : hiddenLayers) {
        output = arma::tanh(layer * output);
    }

    output = arma::tanh(outputLayer * output);
    
    player.right = output(RIGHT) > 0.0f;
    player.left  = output(LEFT)  > 0.0f;
    player.jump  = output(JUMP)  > 0.0f;
}
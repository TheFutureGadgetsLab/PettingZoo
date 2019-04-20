#include <armadillo>
#include <vector>
#include <random>
#include <defs.hpp>
#include <gamelogic.hpp>
#include <NeuralNetwork.hpp>

void split(arma::Mat<float>& parentA, arma::Mat<float>& parentB, arma::Mat<float>& childA, arma::Mat<float>& childB, int splitLoc);

NeuralNetwork::NeuralNetwork(Params params)
{
    fitness = 0;

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

NeuralNetwork::NeuralNetwork(std::string fname)
{
    fitness = 0;

    unsigned int level_seed;
    std::ifstream inFile;

    inFile.open(fname, std::ios::in | std::ios::binary);

    // Level seed
    inFile.read(reinterpret_cast<char*>(&level_seed), sizeof(level_seed));
    
    // Header
    inFile.read(reinterpret_cast<char*>(&inH), sizeof(inH));
    inFile.read(reinterpret_cast<char*>(&inW), sizeof(inW));
    inFile.read(reinterpret_cast<char*>(&hlc), sizeof(hlc));
    inFile.read(reinterpret_cast<char*>(&npl), sizeof(npl));

    // Layers
    inputLayer.load(inFile); 

    hiddenLayers.resize(hlc - 1);
    for (arma::Mat<float>& layer : hiddenLayers) {
        layer.load(inFile); 
    }
    
    outputLayer.load(inFile);

    inFile.close();
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
    printf("-------------------------------------------");
    int r, c, hl;

    printf("\nInput to first hidden layer adj:\n");
    std::cout << inputLayer << std::endl << std::endl;

    for (hl = 0; hl < this->hlc - 1; hl++) {
        printf("Hidden layer %d to %d act:\n", hl + 1, hl + 2);
        std::cout << hiddenLayers[hl] << std::endl << std::endl;
    }

    printf("Hidden layer %d to output act:\n", this->hlc);
    std::cout << outputLayer << std::endl;


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

void NeuralNetwork::mutate(float mutateRate)
{
    if (mutateRate == 0.0f)
        return;
    
    arma::Mat<float> tmpTranspose;

    std::uniform_real_distribution<> weightGenerator(-1.0f, 1.0f);
    std::uniform_real_distribution<> chanceGenerator(0.0f, 1.0f);

    tmpTranspose = inputLayer.t();
    for (float& weight : tmpTranspose) {
        if (chanceGenerator(engine) < mutateRate) {
            weight *= weightGenerator(engine);
        }
    }
    inputLayer = tmpTranspose.t();

    for (arma::Mat<float>& layer : hiddenLayers) {
        tmpTranspose = layer.t();
        for (float& weight : tmpTranspose) {
            if (chanceGenerator(engine) < mutateRate) {
                weight *= weightGenerator(engine);
            }
        }
        layer = tmpTranspose.t();
    }

    tmpTranspose = outputLayer.t();
    for (float& weight : tmpTranspose) {
        if (chanceGenerator(engine) < mutateRate) {
            weight *= weightGenerator(engine);
        }
    }
    outputLayer = tmpTranspose.t();
}

void NeuralNetwork::writeToFile(std::string fname, unsigned int level_seed)
{
    std::ofstream outFile;
    outFile.open(fname, std::ios::out | std::ios::binary);

    // Level seed
    outFile.write(reinterpret_cast<char*>(&level_seed), sizeof(level_seed));

    // Chromosome structure
    outFile.write(reinterpret_cast<char*>(&inH), sizeof(inH)); 
    outFile.write(reinterpret_cast<char*>(&inW), sizeof(inW)); 
    outFile.write(reinterpret_cast<char*>(&hlc), sizeof(hlc)); 
    outFile.write(reinterpret_cast<char*>(&npl), sizeof(npl)); 

    // Layers
    inputLayer.save(outFile); 
    for (arma::Mat<float>& layer : hiddenLayers) {
        layer.save(outFile); 
    }
    outputLayer.save(outFile);
    
    outFile.close();
}

void breed(NeuralNetwork& parentA, NeuralNetwork& parentB, NeuralNetwork& childA, NeuralNetwork& childB, unsigned int seed)
{
    int split_loc, hl;
    unsigned int seedState = seed;

    // Cross input adj layers and mutate
    split_loc = rand_r(&seedState) % (parentA.inputLayer.size() + 1);
    split(parentA.inputLayer, parentB.inputLayer, childA.inputLayer, childB.inputLayer, split_loc);

    // Cross hidden layers and mutate
    for (int layer = 0; layer < parentA.hiddenLayers.size(); layer++) {
        split_loc = rand_r(&seedState) % (parentA.hiddenLayers[layer].size() + 1);
        split(parentA.hiddenLayers[layer], parentB.hiddenLayers[layer], childA.hiddenLayers[layer], childB.hiddenLayers[layer], split_loc);
    }
        
    // Cross output adj layer and mutate
    split_loc = rand_r(&seedState) % (parentA.outputLayer.size() + 1);
    split(parentA.outputLayer, parentB.outputLayer, childA.outputLayer, childB.outputLayer, split_loc);
}

void split(arma::Mat<float>& parentA, arma::Mat<float>& parentB, arma::Mat<float>& childA, arma::Mat<float>& childB, int splitLoc)
{
    arma::Mat<float> pAt = parentA.t();
    arma::Mat<float> pBt = parentB.t();

    childA.reshape(childA.n_cols, childA.n_rows);
    childB.reshape(childB.n_cols, childB.n_rows);

    // Copy split elements of parentA into childA
    std::copy(pAt.begin(), pAt.begin() + splitLoc, childA.begin());
    std::copy(pBt.begin() + splitLoc, pBt.end(), childA.begin() + splitLoc);

    // Copy splitLoc elements of pAt into childA
    std::copy(pBt.begin(), pBt.begin() + splitLoc, childB.begin());
    std::copy(pAt.begin() + splitLoc, pAt.end(), childB.begin() + splitLoc);

    childA = childA.t();
    childB = childB.t();
}

unsigned int getStatsFromFile(std::string fname, Params& params)
{
    unsigned int level_seed;
    std::ifstream data_file;

    data_file.open(fname, std::ios::in | std::ios::binary);

    // Level seed
    data_file.read(reinterpret_cast<char*>(&level_seed), sizeof(level_seed));

    // Chromosome header
    data_file.read(reinterpret_cast<char*>(&params.in_h), sizeof(params.in_h));
    data_file.read(reinterpret_cast<char*>(&params.in_w), sizeof(params.in_w));
    data_file.read(reinterpret_cast<char*>(&params.hlc), sizeof(params.hlc));
    data_file.read(reinterpret_cast<char*>(&params.npl), sizeof(params.npl));

    data_file.close();

    return level_seed;
}
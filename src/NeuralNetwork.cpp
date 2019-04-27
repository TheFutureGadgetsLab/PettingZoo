#include <armadillo>
#include <vector>
#include <random>
#include <defs.hpp>
#include <gamelogic.hpp>
#include <NeuralNetwork.hpp>

void MMIntraNeuronBreed(NeuralNetwork& parentA, NeuralNetwork& parentB, NeuralNetwork& childA, NeuralNetwork& childB, unsigned int seed);
void MMOnNeuronBreed(NeuralNetwork& parentA, NeuralNetwork& parentB, NeuralNetwork& childA, NeuralNetwork& childB, unsigned int seed);
void interpolateBreed(NeuralNetwork& parentA, NeuralNetwork& parentB, NeuralNetwork& childA, NeuralNetwork& childB, unsigned int seed);

void MMIntraSplit(arma::Mat<float>& parentA, arma::Mat<float>& parentB, arma::Mat<float>& childA, arma::Mat<float>& childB, int splitLoc);
void MMOnNeuronSplit(arma::Mat<float>& parentA, arma::Mat<float>& parentB, arma::Mat<float>& childA, arma::Mat<float>& childB, int splitLoc);
void interpolateSplit(arma::Mat<float>& parentA, arma::Mat<float>& parentB, arma::Mat<float>& childA, arma::Mat<float>& childB, float mag);

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
    std::uniform_real_distribution<float> weightGen(-1.0, 1.0);
    
    for (float& val : inputLayer) {
        val = weightGen(engine);
    }

    for (arma::Mat<float>& layer : hiddenLayers) {
        for (float& val : layer) {
            val = weightGen(engine);
        }
    }

    for (float& val : outputLayer) {
        val = weightGen(engine);
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
    arma::Mat<float> input;
    std::vector<float> tiles(inW * inH);

    game.getInputTiles(player, tiles, inH, inW);
    input = arma::Mat<float>(tiles);

    input = arma::tanh(inputLayer * input);
    
    for (arma::Mat<float>& layer : hiddenLayers) {
        input = arma::tanh(layer * input);
    }

    input = arma::tanh(outputLayer * input);
    
    player.right = input(RIGHT) > 0.0f;
    player.left  = input(LEFT)  > 0.0f;
    player.jump  = input(JUMP)  > 0.0f;
}

void NeuralNetwork::mutate(float mutateRate)
{
    if (mutateRate == 0.0f)
        return;
    
    std::uniform_real_distribution<float> weightGen(-1.0f, 1.0f);
    std::uniform_real_distribution<float> chanceGen(0.0f, 1.0f);

    // For each value in the matrix, if random num is < mutateRate then the value is multiplied by a randon number, otherwise it doesnt change
    inputLayer.for_each( [&](float& val) { if (chanceGen(engine) < mutateRate) val *= weightGen(engine); } );
    
    for (arma::Mat<float>& layer : hiddenLayers) {
        layer.for_each( [&](float& val) { if (chanceGen(engine) < mutateRate) val *= weightGen(engine); } );
    }

    outputLayer.for_each( [&](float& val) { if (chanceGen(engine) < mutateRate) val *= weightGen(engine); } );
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

void breed(NeuralNetwork& parentA, NeuralNetwork& parentB, NeuralNetwork& childA, NeuralNetwork& childB, unsigned int seed, int breedType)
{
    if (breedType == 0) {
        MMIntraNeuronBreed(parentA, parentB, childA, childB, seed);
    } else if (breedType == 1) {
        MMOnNeuronBreed(parentA, parentB, childA, childB, seed);
    } else if (breedType == 2) {
        interpolateBreed(parentA, parentB, childA, childB, seed);
    } else {
        printf("Unrecognized breed type!\n");
        exit(-1);
    }
}

void MMIntraNeuronBreed(NeuralNetwork& parentA, NeuralNetwork& parentB, NeuralNetwork& childA, NeuralNetwork& childB, unsigned int seed)
{
    int split_loc, hl;
    unsigned int seedState = seed;

    // Cross input adj layers and mutate
    split_loc = rand_r(&seedState) % (parentA.inputLayer.size() + 1);
    MMIntraSplit(parentA.inputLayer, parentB.inputLayer, childA.inputLayer, childB.inputLayer, split_loc);

    // Cross hidden layers and mutate
    for (int layer = 0; layer < parentA.hiddenLayers.size(); layer++) {
        split_loc = rand_r(&seedState) % (parentA.hiddenLayers[layer].size() + 1);
        MMIntraSplit(parentA.hiddenLayers[layer], parentB.hiddenLayers[layer], childA.hiddenLayers[layer], childB.hiddenLayers[layer], split_loc);
    }
        
    // Cross output adj layer and mutate
    split_loc = rand_r(&seedState) % (parentA.outputLayer.size() + 1);
    MMIntraSplit(parentA.outputLayer, parentB.outputLayer, childA.outputLayer, childB.outputLayer, split_loc);
}

void MMIntraSplit(arma::Mat<float>& parentA, arma::Mat<float>& parentB, arma::Mat<float>& childA, arma::Mat<float>& childB, int splitLoc)
{
    arma::Mat<float> pAt = parentA.t();
    arma::Mat<float> pBt = parentB.t();

    childA.set_size(childA.n_cols, childA.n_rows);
    childB.set_size(childB.n_cols, childB.n_rows);

    // Copy split elements of parentA into childA
    std::copy(pAt.begin(), pAt.begin() + splitLoc, childA.begin());
    std::copy(pBt.begin() + splitLoc, pBt.end(), childA.begin() + splitLoc);

    // Copy splitLoc elements of pAt into childA
    std::copy(pBt.begin(), pBt.begin() + splitLoc, childB.begin());
    std::copy(pAt.begin() + splitLoc, pAt.end(), childB.begin() + splitLoc);

    childA = childA.t();
    childB = childB.t();
}

void MMOnNeuronBreed(NeuralNetwork& parentA, NeuralNetwork& parentB, NeuralNetwork& childA, NeuralNetwork& childB, unsigned int seed)
{
    int splitLoc, hl;
    unsigned int seedState = seed;

    // Cross input adj layers and mutate
    splitLoc = rand_r(&seedState) % (parentA.inputLayer.size() + 1);
    splitLoc = ceilf((float)splitLoc / (float)parentA.inputLayer.n_cols);
    MMOnNeuronSplit(parentA.inputLayer, parentB.inputLayer, childA.inputLayer, childB.inputLayer, splitLoc);

    // Cross hidden layers and mutate
    for (int layer = 0; layer < parentA.hiddenLayers.size(); layer++) {
        splitLoc = rand_r(&seedState) % (parentA.hiddenLayers[layer].n_elem + 1);
        splitLoc = ceilf((float)splitLoc / (float)parentA.hiddenLayers[layer].n_cols);
        MMOnNeuronSplit(parentA.hiddenLayers[layer], parentB.hiddenLayers[layer], childA.hiddenLayers[layer], childB.hiddenLayers[layer], splitLoc);
    }
        
    // Cross output adj layer and mutate
    splitLoc = rand_r(&seedState) % (parentA.outputLayer.n_elem + 1);
    splitLoc = ceilf((float)splitLoc / (float)parentA.outputLayer.n_cols);
    MMOnNeuronSplit(parentA.outputLayer, parentB.outputLayer, childA.outputLayer, childB.outputLayer, splitLoc);
}

void MMOnNeuronSplit(arma::Mat<float>& parentA, arma::Mat<float>& parentB, arma::Mat<float>& childA, arma::Mat<float>& childB, int splitLoc)
{
    if (splitLoc == parentA.n_rows) {
        childA = parentA;
        childB = parentB;
        return;
    } else if (splitLoc == 0) {
        childA = parentB;
        childB = parentA;
        return;
    }

    childA = arma::join_vert(parentA.rows(0, splitLoc - 1), parentB.rows(splitLoc, parentB.n_rows - 1));
    childB = arma::join_vert(parentB.rows(0, splitLoc - 1), parentA.rows(splitLoc, parentA.n_rows - 1));
}

void interpolateBreed(NeuralNetwork& parentA, NeuralNetwork& parentB, NeuralNetwork& childA, NeuralNetwork& childB, unsigned int seed)
{
    std::uniform_real_distribution<float> magnitude(0.0f, 1.0f);
    std::minstd_rand engine(seed);
    float mag;

    mag = magnitude(engine);
    interpolateSplit(parentA.inputLayer, parentB.inputLayer, childA.inputLayer, childB.inputLayer, mag);

    // Cross hidden layers and mutate
    for (int layer = 0; layer < parentA.hiddenLayers.size(); layer++) {
        mag = magnitude(engine);
        interpolateSplit(parentA.hiddenLayers[layer], parentB.hiddenLayers[layer], childA.hiddenLayers[layer], childB.hiddenLayers[layer], mag);
    }
        
    // Cross output adj layer and mutate
    mag = magnitude(engine);
    interpolateSplit(parentA.outputLayer, parentB.outputLayer, childA.outputLayer, childB.outputLayer, mag);
}

void interpolateSplit(arma::Mat<float>& parentA, arma::Mat<float>& parentB, arma::Mat<float>& childA, arma::Mat<float>& childB, float mag)
{
    childA = (parentA + parentB) * mag;
    childB = (parentA + parentB) * (1 - mag);
}
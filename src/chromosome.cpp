/**
 * @file chromosome.cpp
 * @author Benjamin Mastripolito, Haydn Jones
 * @brief Functions for interfacing with chromosomes
 * @date 2018-12-06
 */

#include <stdio.h>
#include <stdlib.h>
#include <chromosome.hpp>
#include <defs.hpp>
#include <sys/stat.h>
#include <string.h>
#include <algorithm>
#include <fstream>

float gen_random_weight(unsigned int *seedp);

///////////////////////////////////////////////////////////////
//
//                  Chromosome Class
//
///////////////////////////////////////////////////////////////
Chromosome::Chromosome(const char *fname)
{
    unsigned int level_seed;
    std::ifstream data_file;

    data_file.open(fname, std::ios::in | std::ios::binary);

    // Level seed
    data_file.read(reinterpret_cast<char*>(&level_seed), sizeof(level_seed));

    // Chromosome header
    data_file.read(reinterpret_cast<char*>(&in_h), sizeof(in_h));
    data_file.read(reinterpret_cast<char*>(&in_w), sizeof(in_w));
    data_file.read(reinterpret_cast<char*>(&hlc), sizeof(hlc));
    data_file.read(reinterpret_cast<char*>(&npl), sizeof(npl));

    input_adj_size = (in_h * in_w) * npl;
    hidden_adj_size = (hlc - 1) * (npl * npl);
    out_adj_size = BUTTON_COUNT * npl;

    input_adj.resize(input_adj_size);
    
    hiddenLayers.resize(hlc - 1);
    for (int i = 0; i < hiddenLayers.size(); i++) {
        hiddenLayers[i].resize(npl * npl);
    }

    out_adj.resize(out_adj_size);

    input_tiles.resize(in_w * in_h);
    node_outputs.resize(npl * hlc);


    // Matrices    
    data_file.read(reinterpret_cast<char*>(&input_adj[0]), input_adj_size * sizeof(float));

    for (std::vector<float>& layer : hiddenLayers) {
        data_file.read(reinterpret_cast<char*>(&layer[0]), layer.size() * sizeof(float));
    }

    data_file.read(reinterpret_cast<char*>(&out_adj[0]), out_adj_size * sizeof(float));

    data_file.close();
}

/**
 * @brief Initializes and allocates a given chromosome with the given size
 * 
 * This allocates memory and the chromosome MUST be freed with free_chromosome(...)
 * 
 * @param params Parameters to use in chromosome initialization
 */
Chromosome::Chromosome(Params& params)
{
    in_h = params.in_h;
    in_w = params.in_w;
    hlc = params.hlc;
    npl = params.npl;

    input_adj_size = (in_h * in_w) * npl;
    hidden_adj_size = (hlc - 1) * (npl * npl);
    out_adj_size = BUTTON_COUNT * npl;

    input_adj.resize(input_adj_size);
    
    hiddenLayers.resize(hlc - 1);
    for (int i = 0; i < hiddenLayers.size(); i++) {
        hiddenLayers[i].resize(npl * npl);
    }

    out_adj.resize(out_adj_size);

    input_tiles.resize(in_w * in_h);
    node_outputs.resize(npl * hlc);
}

/**
 * @brief Generates a random chromosome using the given seed (must have been initialized first)
 * 
 * @param chrom Chromosome to store weights in
 * @param seed used to seed random number generator
 */
void Chromosome::generate(unsigned int seed)
{
    unsigned int seedp;

    seedp = seed;

    for (int weight = 0; weight < input_adj_size; weight++) {
        input_adj[weight] = gen_random_weight(&seedp);
    }

    for (int layer = 0; layer < hiddenLayers.size(); layer++) {
        for (int weight = 0; weight < hiddenLayers[layer].size(); weight++) {
            hiddenLayers[layer][weight] = gen_random_weight(&seedp);
        }
    }

    for (int weight = 0; weight < out_adj_size; weight++) {
        out_adj[weight] = gen_random_weight(&seedp);
    }
}

/**
 * @brief Prints information and properties of the given chromosome
 * 
 * @param chrom the chromsome to print the properties of
 */
void Chromosome::print()
{
    printf("-------------------------------------------\n");
    int r, c, hl;

    printf("\nInput to first hidden layer adj:\n");
    for (r = 0; r < this->npl; r++) {
        for (c = 0; c < this->in_h * this->in_w; c++) {
            printf("%*.3lf\t", 6, this->input_adj[r * this->in_h * this->in_w + c]);
        }
        puts("");
    }

    puts("");
    for (hl = 0; hl < this->hlc - 1; hl++) {
        printf("Hidden layer %d to %d act:\n", hl + 1, hl + 2);
        for (r = 0; r < this->npl; r++) {
            for (c = 0; c < this->npl; c++) {
                printf("%*.3lf\t", 6, this->hiddenLayers[hl][r * this->npl + c]);
            }
            puts("");
        }
        puts("");
    }

    printf("Hidden layer %d to output act:\n", this->hlc);
    for (r = 0; r < BUTTON_COUNT; r++) {
        for (c = 0; c < this->npl; c++) {
            printf("%*.3lf\t", 6, this->out_adj[r * this->npl + c]);
        }
        puts("");
    }

    printf("\nChromosome:\n");
    printf("in_w:\t%d\nin_h:\t%d\nnpl:\t%d\nhlc:\t%d\n", this->in_h, this->in_w, this->npl, this->hlc);
    printf("Size: %ld bytes\n", (this->input_adj_size + this->hidden_adj_size + this->out_adj_size) * sizeof(float));
    printf("-------------------------------------------\n");
}

/**
 * @brief writes a given chromosome (and seed) to a file. Level seed is the level it was tested on
 * 
 * @param fname file to write to
 * @param chrom the chromosome to write to disk from
 * @param level_seed the seed of the level, which will be written along with the chromosome data
 */
void Chromosome::writeToFile(char *fname, unsigned int level_seed)
{
    std::ofstream data_file;
    data_file.open(fname, std::ios::out | std::ios::binary);

    // Level seed
    data_file.write(reinterpret_cast<char*>(&level_seed), sizeof(float));

    // Chromosome structure
    data_file.write(reinterpret_cast<char*>(&this->in_h), sizeof(this->in_h)); 
    data_file.write(reinterpret_cast<char*>(&this->in_w), sizeof(this->in_w)); 
    data_file.write(reinterpret_cast<char*>(&this->hlc), sizeof(this->hlc)); 
    data_file.write(reinterpret_cast<char*>(&this->npl), sizeof(this->npl)); 

    // Layers
    data_file.write(reinterpret_cast<char*>(&this->input_adj[0]), this->input_adj.size()*sizeof(float)); 
    for (std::vector<float>& layer : this->hiddenLayers) {
        data_file.write(reinterpret_cast<char*>(&layer[0]), layer.size()*sizeof(float)); 
    }
    data_file.write(reinterpret_cast<char*>(&this->out_adj[0]), this->out_adj.size()*sizeof(float)); 

    data_file.close();
}

/**
 * @brief Extracts chromosome from file. IT WILL BE INITIALIZED FOR YOU
 * 
 * @param fname name of file chromosome is stored in
 * @param chrom chromosome obj to fill
 * @return unsigned int 
 */
unsigned int getStatsFromFile(const char *fname, Params& params)
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
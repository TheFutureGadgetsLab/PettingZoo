/**
 * @file chromosome.hpp
 * @author Haydn Jones, Benjamin Mastripolito
 * @brief Holds chromosome function defs and class
 * @date 2018-12-06
 */
#ifndef CHROMOSOME_H
#define CHROMOSOME_H

#include <stdlib.h>
#include <defs.hpp>
#include <vector>
#include <random>

class Chromosome {
    public:
    std::vector<float> inputLayer;                 // Adjacency matrix describing input layer to first hidden layer
    std::vector<std::vector<float>> hiddenLayers; // Adjacency matrix describing the hidden layers
    std::vector<float> outputLayer;                   // Adjacency matrix describing last hidden layer to the output nodes
    std::minstd_rand engine;

    int npl; // Nodes in each hidden layer
    int in_w; // Width of input rectangle around player
    int in_h; // Height of input rectangle around player
    int hlc;  // Number of hidden layers

    Chromosome(Params&);
    Chromosome(const char*);

    void seed(unsigned int);
    void generate();
    void print();
    void writeToFile(char *fname, unsigned int level_seed);
};

unsigned int extract_from_file(const char *fname, Chromosome *chrom);
unsigned int getStatsFromFile(const char *fname, Params& params);
void single_point_breed(Chromosome& parentA, Chromosome& parentB, Chromosome& childA, Chromosome& childB, Params& params, unsigned int seed);

/*

All matrices in the chromosome are flattened, 2D, row-major matrices.

INPUT ADJACENCY MATRIX
The input adjacency matrix is of size (NPL, IN_H * IN_W) describing the weights connecting the 
input nodes to the first hidden layer. The adjacency matrix is transposed to make it 
cache-friendly. Rather than each row describing the connections of a given input tile, it
describes the connections to a given node in the hidden layer.

HIDDEN ADJACENCY MATRIX
The next (HLC - 1) chunks will be adjacency matrices of size (NPL, NPL) describing weights from
the previous hidden layer to the current. Once again, this is logically transposed to be cache-friendly.
Each row represents a node and the columns are connections to it.

OUTPUT ADJACENCY MATRIX
The final chunk will be an adjacency matrix of size (NPL, BUTTON_COUNT) describing the weights
between the final hidden layer and the output layer.Same format as other adjacency matrices.
Note on adjacency matrices:

*/

#endif

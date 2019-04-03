/**
 * @file chromosome.hpp
 * @author Haydn Jones, Benjamin Mastripolito
 * @brief Holds chromosome function defs and class
 * @date 2018-12-06
 */
#ifndef CHROMOSOME_H
#define CHROMOSOME_H

#include <stdint.h>
#include <stdlib.h>
#include <defs.hpp>
#include <vector>

class Chromosome {
    public:
    std::vector<float> input_adj;  // Adjacency matrix describing input layer to first hidden layer
    std::vector<float> hidden_adj; // Adjacency matrix describing the hidden layers
    std::vector<float> out_adj;    // Adjacency matrix describing last hidden layer to the output nodes
    std::vector<float> input_tiles;
    std::vector<float> node_outputs;
    
    // Sizes are number of elements, not bytes
    size_t input_adj_size;
    size_t hidden_adj_size;
    size_t out_adj_size;

    uint16_t npl; // Nodes in each hidden layer
    uint8_t in_w; // Width of input rectangle around player
    uint8_t in_h; // Height of input rectangle around player
    uint8_t hlc;  // Number of hidden layers

    Chromosome(Params&);
    Chromosome(const char*);
    Chromosome(const Chromosome &); // Copy constructor
    ~Chromosome();
    void generate(unsigned int);
};

void print_chromosome(Chromosome& chrom);
void write_out_chromosome(char *fname, Chromosome& chrom, unsigned int level_seed);
unsigned int extract_from_file(const char *fname, Chromosome *chrom);
unsigned int getStatsFromFile(const char *fname, Params& params);

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

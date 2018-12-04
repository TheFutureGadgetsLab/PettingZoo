#ifndef CHROMOSOME_H
#define CHROMOSOME_H

#include <stdint.h>
#include <stdlib.h>

struct Chromosome {
    float *input_adj;
    float *hidden_adj;
    float *out_adj;
    
    // Sizes are number of elements, not bytes
    size_t input_adj_size;
    size_t hidden_adj_size;
    size_t out_adj_size;

    uint16_t npl;
    uint8_t in_w;
    uint8_t in_h;
    uint8_t hlc;
};

void free_chromosome(struct Chromosome *chrom);
void initialize_chromosome(struct Chromosome *chrom, uint8_t in_h, uint8_t in_w, uint8_t hlc, uint16_t npl);
void generate_chromosome(struct Chromosome *chrom, unsigned int seed);
void print_chromosome(struct Chromosome *chrom);
void write_out_chromosome(char *fname, struct Chromosome *chrom, unsigned int level_seed);
unsigned int extract_from_file(const char *fname, struct Chromosome *chrom);

/*

All matrices in the chromosome will be flattened, 2D, row-major matrices with elements of type
uint8_t unless otherwise stated.

The parameters that define the size of a chromosome (notably nodes per hidden layer and hidden
layer count) will likely change so we will embed the size parameters in the beginning of a
chcromosome to allow some amount of portability. The first elements of a chromosome will be as
follows (byte indexed):

    0:   IN_W (input width)
    1:   IN_H (input height)
    2-3: NPL (nodes per hidden layer)
    4:   HLC (hidden layer count)


INPUT ACTIVATION MATRIX
The first chunk of the chromosome will be a matrix of size (IN_H, IN_W) describing which
input tiles are active. The array is arranged from top to bottom, left to right w.r.t. the 
input (top left tile of input is first index, bottom right tile is last index). It currently 
makes sense to me that an inactive input tile should report that there is simply nothing there 
(empty tile). A 0 in this matrix at index i,j means that input tile i,j is inactive. A 1 means 
the input is active.

INPUT ADJACENCY MATRIX
The next chunk will be an adjacency matrix of size (NPL, IN_H * IN_W) describing the weights 
connecting the input nodes to the first hidden layer. Each element will be of type float. The
adjacency matrix is transposed to make it cache-friendly. Rather than (A)_ij representing a 
connection from i to j, it represents a connection from j to i, where j is a linear index into
the input tile array. Outputs of nodes are calculated one by one, so placing all the connections
to a given node in a row, rather than a column makes sense.

HIDDEN ACTIVATION MATRIX
The next chunk will be a matrix of size (HLC, NPL) describing which neurons are active in the
hidden layers. A 0 means active, and a 1 means inactive. This is transposed ((HLC, NPL) rather
than (NPL, HLC), which is the "logical view") to be cache friendly. Each row corresponds to an
entire layer.

HIDDEN ADJACENCY MATRIX
The next (HLC - 1) chunks will be adjacency matrices of size (NPL, NPL) describing weights
between the previous hidden layer and the current. Each element will be of type float. Once
again, this is logically transposed for cache-friendlyness. Each row represents a node and the 
columns are connections to it.

OUTPUT ADJACENCY MATRIX
The final chunk will be an adjacency matrix of size (NPL, BUTTON_COUNT) describing the weights
between the final hidden layer and the output layer. Each element will be of type float. Same
format as other adjacency matrices, each row represents a button and the columns are connections
to that button.

Note on adjacency matrices:
Connections will be "disabled" by setting the relevant weight to 0.

*/

#endif

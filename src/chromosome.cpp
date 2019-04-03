/**
 * @file chromosome.cpp
 * @author Benjamin Mastripolito, Haydn Jones
 * @brief Functions for interfacing with chromosomes
 * @date 2018-12-06
 */

#include <stdio.h>
#include <stdlib.h>
#include <chromosome.hpp>
#include <stdint.h>
#include <defs.hpp>
#include <sys/stat.h>
#include <string.h>
#include <algorithm>

float gen_random_weight(unsigned int *seedp);

///////////////////////////////////////////////////////////////
//
//                  Chromosome Class
//
///////////////////////////////////////////////////////////////
/**
 * @brief Initializes and allocates a given chromosome with the given size
 * 
 * This allocates memory and the chromosome MUST be freed with free_chromosome(...)
 * 
 * @param chrom The chromsome to initialize
 * @param in_w Input matrix width
 * @param in_h Input matrix height
 * @param hlc Hidden layer count
 * @param npl Nodes per layer
 */
Chromosome::Chromosome(const char *fname)
{
    FILE *file = NULL;
    struct stat st;
    size_t read;
    unsigned int level_seed;

    if (stat(fname, &st) == -1) {
		printf("Error reading file '%s'!\n", fname);
		exit(EXIT_FAILURE);
	}

    file = fopen(fname, "rb");

    // Level seed
    read = fread(&level_seed, sizeof(level_seed), 1, file);

    // Chromosome header
    read = fread(&in_h, sizeof(in_h), 1, file);
    read = fread(&in_w, sizeof(in_w), 1, file);
    read = fread(&hlc, sizeof(hlc), 1, file);
    read = fread(&npl, sizeof(npl), 1, file);

    input_adj_size = (in_h * in_w) * npl;
    hidden_adj_size = (hlc - 1) * (npl * npl);
    out_adj_size = BUTTON_COUNT * npl;

    input_adj = new float[input_adj_size];
    hidden_adj = new float[hidden_adj_size];
    out_adj = new float[out_adj_size];

    input_tiles = new float[in_w * in_h];
    node_outputs = new float[npl * hlc];

    // Matrices       
    read = fread(input_adj, sizeof(float), input_adj_size, file);
    read = fread(hidden_adj, sizeof(float), hidden_adj_size, file);
    read = fread(out_adj, sizeof(float), out_adj_size, file);

    fclose(file);
}

Chromosome::Chromosome(Params& params)
{
    in_h = params.in_h;
    in_w = params.in_w;
    hlc = params.hlc;
    npl = params.npl;

    input_adj_size = (in_h * in_w) * npl;
    hidden_adj_size = (hlc - 1) * (npl * npl);
    out_adj_size = BUTTON_COUNT * npl;

    input_adj = new float[input_adj_size];
    hidden_adj = new float[hidden_adj_size];
    out_adj = new float[out_adj_size];

    input_tiles = new float[in_w * in_h];
    node_outputs = new float[npl * hlc];
}

// Copy Constructor
Chromosome::Chromosome(const Chromosome& old)
{
    in_h = old.in_h;
    in_w = old.in_w;
    hlc = old.hlc;
    npl = old.npl;

    input_adj_size = (in_h * in_w) * npl;
    hidden_adj_size = (hlc - 1) * (npl * npl);
    out_adj_size = BUTTON_COUNT * npl;

    input_adj = new float[input_adj_size];
    hidden_adj = new float[hidden_adj_size];
    out_adj = new float[out_adj_size];

    input_tiles = new float[in_w * in_h];
    node_outputs = new float[npl * hlc];

    // Copy over NN
    memcpy(input_adj, old.input_adj, input_adj_size * sizeof(float));
    memcpy(hidden_adj, old.hidden_adj, hidden_adj_size * sizeof(float));
    memcpy(out_adj, old.out_adj, out_adj_size * sizeof(float));

    memcpy(input_tiles, old.input_tiles, in_w * in_h * sizeof(float));
    memcpy(node_outputs, old.node_outputs, npl * hlc * sizeof(float));
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

    for (int weight = 0; weight < hidden_adj_size; weight++) {
        hidden_adj[weight] = gen_random_weight(&seedp);
    }

    for (int weight = 0; weight < out_adj_size; weight++) {
        out_adj[weight] = gen_random_weight(&seedp);
    }
}

/**
 * @brief Frees the memory used by chrom
 * 
 * @param chrom chromsome to free
 */
Chromosome::~Chromosome()
{
    delete [] input_adj;
    delete [] hidden_adj;
    delete [] out_adj;

    delete [] input_tiles;
    delete [] node_outputs;
}

/**
 * @brief Prints information and properties of the given chromosome
 * 
 * @param chrom the chromsome to print the properties of
 */
void print_chromosome(Chromosome *chrom)
{
    printf("-------------------------------------------\n");
    uint8_t *cur_uint;
    float *cur_float;
    int r, c, hl;

    printf("\nInput to first hidden layer adj:\n");
    cur_float = chrom->input_adj;
    for (r = 0; r < chrom->npl; r++) {
        for (c = 0; c < chrom->in_h * chrom->in_w; c++) {
            printf("%*.3lf\t", 6, *cur_float);
            cur_float++;
        }
        puts("");
    }

    puts("");
    cur_float = chrom->hidden_adj;
    for (hl = 0; hl < chrom->hlc - 1; hl++) {
        printf("Hidden layer %d to %d act:\n", hl + 1, hl + 2);
        for (r = 0; r < chrom->npl; r++) {
            for (c = 0; c < chrom->npl; c++) {
                printf("%*.3lf\t", 6, *cur_float);
                cur_float++;
            }
            puts("");
        }
        puts("");
    }

    printf("Hidden layer %d to output act:\n", chrom->hlc);
    cur_float = chrom->out_adj;
    for (r = 0; r < BUTTON_COUNT; r++) {
        for (c = 0; c < chrom->npl; c++) {
            printf("%*.3lf\t", 6, *cur_float);
            cur_float++;
        }
        puts("");
    }

    printf("\nChromosome:\n");
    printf("in_w:\t%d\nin_h:\t%d\nnpl:\t%d\nhlc:\t%d\n", chrom->in_h, chrom->in_w, chrom->npl, chrom->hlc);
    printf("Size: %ld bytes\n", (chrom->input_adj_size + chrom->hidden_adj_size + chrom->out_adj_size) * sizeof(float));
    printf("-------------------------------------------\n");
}

/**
 * @brief Generate a random weight in range [-1, 1]
 * 
 * @param seedp random number generator seed
 * @return float number between [-1, 1]
 */
float gen_random_weight(unsigned int *seedp)
{
    float weight, chance;

    weight = (float)rand_r(seedp) / RAND_MAX;

    // Flip sign on even
    if (rand_r(seedp) % 2 == 0)
        weight = weight * -1;

    return weight;
}

/**
 * @brief writes a given chromosome (and seed) to a file. Level seed is the level it was tested on
 * 
 * @param fname file to write to
 * @param chrom the chromosome to write to disk from
 * @param level_seed the seed of the level, which will be written along with the chromosome data
 */
void write_out_chromosome(char *fname, Chromosome& chrom, unsigned int level_seed)
{
    FILE *file = fopen(fname, "wb");

    // Level seed
    fwrite(&level_seed, sizeof(level_seed), 1, file);

    //Chromosome
    fwrite(&chrom.in_h, sizeof(chrom.in_h), 1, file);
    fwrite(&chrom.in_w, sizeof(chrom.in_w), 1, file);
    fwrite(&chrom.hlc, sizeof(chrom.hlc), 1, file);
    fwrite(&chrom.npl, sizeof(chrom.npl), 1, file);

    fwrite(chrom.input_adj, sizeof(*chrom.input_adj), chrom.input_adj_size, file);
    fwrite(chrom.hidden_adj, sizeof(*chrom.hidden_adj), chrom.hidden_adj_size, file);
    fwrite(chrom.out_adj, sizeof(*chrom.out_adj), chrom.out_adj_size, file);

    fclose(file);
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
    FILE *file = NULL;
    struct stat st;
    size_t read;
    unsigned int level_seed;


    if (stat(fname, &st) == -1) {
		printf("Error reading file '%s'!\n", fname);
		exit(EXIT_FAILURE);
	}

    file = fopen(fname, "rb");
    
    // Level seed
    read = fread(&level_seed, sizeof(level_seed), 1, file);

    // Chromosome header
    read = fread(&params.in_h, sizeof(params.in_h), 1, file);
    read = fread(&params.in_w, sizeof(params.in_w), 1, file);
    read = fread(&params.hlc, sizeof(params.hlc), 1, file);
    read = fread(&params.npl, sizeof(params.npl), 1, file);

    fclose(file);

    return level_seed;
}
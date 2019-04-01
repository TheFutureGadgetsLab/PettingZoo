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

float gen_random_weight(unsigned int *seedp);

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
void initialize_chromosome(Chromosome *chrom, uint8_t in_w, uint8_t in_h, uint8_t hlc, uint16_t npl)
{
    chrom->in_h = in_h;
    chrom->in_w = in_w;
    chrom->hlc = hlc;
    chrom->npl = npl;

    chrom->input_adj_size = (in_h * in_w) * npl;
    chrom->hidden_adj_size = (hlc - 1) * (npl * npl);
    chrom->out_adj_size = BUTTON_COUNT * npl;

    chrom->input_adj = (float *)calloc(chrom->input_adj_size, sizeof(float));
    chrom->hidden_adj = (float *)calloc(chrom->hidden_adj_size, sizeof(float));
    chrom->out_adj = (float *)calloc(chrom->out_adj_size, sizeof(float));
}

/**
 * @brief Frees the memory used by chrom
 * 
 * @param chrom chromsome to free
 */
void free_chromosome(Chromosome *chrom)
{
    free(chrom->input_adj);
    free(chrom->hidden_adj);
    free(chrom->out_adj);
}

/**
 * @brief Generates a random chromosome using the given seed (must have been initialized first)
 * 
 * @param chrom Chromosome to store weights in
 * @param seed used to seed random number generator
 */
void generate_chromosome(Chromosome *chrom, unsigned int seed)
{
    uint8_t *cur_uint;
    float *cur_float;
    int r, c, hl;
    unsigned int seedp;

    seedp = seed;

    // Generate input adj matrix
    cur_float = chrom->input_adj;
    for (r = 0; r < chrom->npl; r++) {
        for (c = 0; c < chrom->in_h * chrom->in_w; c++) {
            *cur_float = gen_random_weight(&seedp);
            cur_float++;
        }
    }

    // Generate hidden adj matrices
    cur_float = chrom->hidden_adj;
    for (hl = 0; hl < chrom->hlc - 1; hl++) {
        for (r = 0; r < chrom->npl; r++) {
            for (c = 0; c < chrom->npl; c++) {
                *cur_float = gen_random_weight(&seedp);
                cur_float++;
            }
        }
    }

    // Generate out adj matrix
    cur_float = chrom->out_adj;
    for (r = 0; r < BUTTON_COUNT; r++) {
        for (c = 0; c < chrom->npl; c++) {
            *cur_float = gen_random_weight(&seedp);
            cur_float++;
        }
    }
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
void write_out_chromosome(char *fname, Chromosome *chrom, unsigned int level_seed)
{
    FILE *file = fopen(fname, "wb");

    // Level seed
    fwrite(&level_seed, sizeof(level_seed), 1, file);

    //Chromosome
    fwrite(&chrom->in_h, sizeof(chrom->in_h), 1, file);
    fwrite(&chrom->in_w, sizeof(chrom->in_w), 1, file);
    fwrite(&chrom->hlc, sizeof(chrom->hlc), 1, file);
    fwrite(&chrom->npl, sizeof(chrom->npl), 1, file);

    fwrite(chrom->input_adj, sizeof(*chrom->input_adj), chrom->input_adj_size, file);
    fwrite(chrom->hidden_adj, sizeof(*chrom->hidden_adj), chrom->hidden_adj_size, file);
    fwrite(chrom->out_adj, sizeof(*chrom->out_adj), chrom->out_adj_size, file);

    fclose(file);
}

/**
 * @brief Extracts chromosome from file. IT WILL BE INITIALIZED FOR YOU
 * 
 * @param fname name of file chromosome is stored in
 * @param chrom chromosome obj to fill
 * @return unsigned int 
 */
unsigned int extract_from_file(const char *fname, Chromosome *chrom)
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
    read = fread(&chrom->in_h, sizeof(chrom->in_h), 1, file);
    read = fread(&chrom->in_w, sizeof(chrom->in_w), 1, file);
    read = fread(&chrom->hlc, sizeof(chrom->hlc), 1, file);
    read = fread(&chrom->npl, sizeof(chrom->npl), 1, file);

    initialize_chromosome(chrom, chrom->in_h, chrom->in_w, chrom->hlc, chrom->npl);

    // Matrices         
    read = fread(chrom->input_adj, sizeof(*chrom->input_adj), chrom->input_adj_size, file);
    read = fread(chrom->hidden_adj, sizeof(*chrom->hidden_adj), chrom->hidden_adj_size, file);
    read = fread(chrom->out_adj, sizeof(*chrom->out_adj), chrom->out_adj_size, file);

    fclose(file);

    return level_seed;
}
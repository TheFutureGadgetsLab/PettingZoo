#include <stdio.h>
#include <stdlib.h>
#include <chromosome.hpp>
#include <stdint.h>
#include <defs.hpp>

float gen_random_weight(unsigned int *seedp);

void initialize_chromosome(struct Chromosome *chrom, uint8_t in_h, uint8_t in_w, uint8_t hlc, uint16_t npl)
{
    chrom->in_h = in_h;
    chrom->in_w = in_w;
    chrom->hlc = hlc;
    chrom->npl = npl;

    chrom->input_act = NULL;
    chrom->input_adj = NULL;
    chrom->hidden_act = NULL;
    chrom->hidden_adj = NULL;
    chrom->out_adj = NULL;

    chrom->input_act_size = in_h * in_w;
    chrom->input_adj_size = (in_h * in_w) * npl;
    chrom->hidden_act_size = (hlc * npl);
    chrom->hidden_adj_size = (hlc - 1) * (npl * npl);
    chrom->out_adj_size = BUTTON_COUNT * npl;

    chrom->input_act = (uint8_t *)malloc(sizeof(uint8_t) * chrom->input_act_size);
    chrom->input_adj = (float *)malloc(sizeof(float) * chrom->input_adj_size);
    chrom->hidden_act = (uint8_t *)malloc(sizeof(uint8_t) * chrom->hidden_act_size);
    chrom->hidden_adj = (float *)malloc(sizeof(float) * chrom->hidden_adj_size);
    chrom->out_adj = (float *)malloc(sizeof(float) * chrom->out_adj_size);
}

void free_chromosome(struct Chromosome *chrom)
{
    free(chrom->input_act);
    free(chrom->input_adj);
    free(chrom->hidden_act);
    free(chrom->hidden_adj);
    free(chrom->out_adj);
}

void generate_chromosome(struct Chromosome *chrom, unsigned int seed)
{
    uint8_t *cur_uint;
    float *cur_float;
    int r, c, hl;
    unsigned int seedp;

    seedp = seed;

    cur_uint = chrom->input_act;
    // Generate input act matrix
    // This is NOT transposed
    for (r = 0; r < chrom->in_h; r++) {
        for (c = 0; c < chrom->in_w; c++) {
            // *cur_uint = rand_r(&seedp) % 2;
            *cur_uint = 1;
            cur_uint++;
        }
    }

    // Generate input adj matrix
    cur_float = chrom->input_adj;
    for (r = 0; r < chrom->npl; r++) {
        for (c = 0; c < chrom->in_h * chrom->in_w; c++) {
            *cur_float = gen_random_weight(&seedp);
            cur_float++;
        }
    }

    // Generate hidden act matrix
    cur_uint = chrom->hidden_act;
    for (r = 0; r < chrom->hlc; r++) {
        for (c = 0; c < chrom->npl; c++) {
            // *cur_uint = rand_r(&seedp) % 2;
            *cur_uint = 1;
            cur_uint++;
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

void print_chromosome(struct Chromosome *chrom)
{
    printf("-------------------------------------------\n");
    uint8_t *cur_uint;
    float *cur_float;
    int r, c, hl;

    printf("Input activation:\n");
    cur_uint = chrom->input_act;
    for (r = 0; r < chrom->in_h; r++) {
        for (c = 0; c < chrom->in_w; c++) {
            printf("%d\t", *cur_uint);
            cur_uint++;
        }
        puts("");
    }

    printf("\nInput to first hidden layer adj:\n");
    cur_float = chrom->input_adj;
    for (r = 0; r < chrom->npl; r++) {
        for (c = 0; c < chrom->in_h * chrom->in_w; c++) {
            printf("%*.3lf\t", 6, *cur_float);
            cur_float++;
        }
        puts("");
    }

    printf("\nHidden layers activation:\n");
    cur_uint = chrom->hidden_act;
    for (r = 0; r < chrom->hlc; r++) {
        for (c = 0; c < chrom->npl; c++) {
            printf("%d\t", *cur_uint);
            cur_uint++;
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
    printf("-------------------------------------------\n");
}

size_t get_chromosome_size_params(uint8_t in_h, uint8_t in_w, uint8_t hlc, uint16_t npl)
{
    size_t size;

    // Algebra to reduce length of this line
    size = 5 + in_h * (in_w + 4 * in_w * npl) + npl * (4 * npl * (-1 + hlc) + hlc + 4 * BUTTON_COUNT);

    return size;
}

float gen_random_weight(unsigned int *seedp)
{
    float weight, chance;
    
    weight = (float)rand_r(seedp) / RAND_MAX;
    chance = (float)rand_r(seedp) / RAND_MAX;
    
    // Connection is inactive
    // THIS IS CURRENTLY NOT IN USE
    if (chance > 2.0f)
        return 0.0f;

    // Flip sign on even
    if (rand_r(seedp) % 2 == 0)
        weight = weight * -1;

    return weight;
}


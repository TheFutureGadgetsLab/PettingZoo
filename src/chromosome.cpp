#include <stdio.h>
#include <stdlib.h>
#include <chromosome.hpp>
#include <stdint.h>
#include <defs.hpp>

float gen_random_weight(unsigned int *seedp);

void generate_chromosome(uint8_t *chrom, uint8_t in_h, uint8_t in_w, uint8_t hlc, uint16_t npl, unsigned int seed)
{
    uint8_t *cur_uint;
    float *cur_float;
    int r, c, hl;
    unsigned int seedp;
    struct params params;

    seedp = seed;

    chrom[0] = in_w;
    chrom[1] = in_h;
    *((uint16_t *)chrom + 1) = npl; // Bytes 2-3
    chrom[4] = hlc;
    
    get_params(chrom, &params);

    cur_uint = params.input_act;
    // Generate input act matrix
    // This is NOT transposed
    for (r = 0; r < in_h; r++) {
        for (c = 0; c < in_w; c++) {
            // *cur_uint = rand_r(&seedp) % 2;
            *cur_uint = 1;
            cur_uint++;
        }
    }

    // Generate input adj matrix
    cur_float = params.input_adj;
    for (r = 0; r < npl; r++) {
        for (c = 0; c < in_h * in_w; c++) {
            *cur_float = gen_random_weight(&seedp);
            cur_float++;
        }
    }

    // Generate hidden act matrix
    cur_uint = params.hidden_act;
    for (r = 0; r < hlc; r++) {
        for (c = 0; c < npl; c++) {
            // *cur_uint = rand_r(&seedp) % 2;
            *cur_uint = 1;
            cur_uint++;
        }
    }

    // Generate hidden adj matrices
    cur_float = params.hidden_adj;
    for (hl = 0; hl < hlc - 1; hl++) {
        for (r = 0; r < npl; r++) {
            for (c = 0; c < npl; c++) {
                *cur_float = gen_random_weight(&seedp);
                cur_float++;
            }
        }
    }

    // Generate out adj matrix
    cur_float = params.out_adj;
    for (r = 0; r < BUTTON_COUNT; r++) {
        for (c = 0; c < npl; c++) {
            *cur_float = gen_random_weight(&seedp);
            cur_float++;
        }
    }
}

void print_chromosome(uint8_t *chrom)
{
    printf("-------------------------------------------\n");
    uint8_t *cur_uint;
    float *cur_float;
    int r, c, hl;
    struct params params;

    get_params(chrom, &params);

    printf("Input activation:\n");
    cur_uint = params.input_act;
    for (r = 0; r < params.in_h; r++) {
        for (c = 0; c < params.in_w; c++) {
            printf("%d\t", *cur_uint);
            cur_uint++;
        }
        puts("");
    }

    printf("\nInput to first hidden layer adj:\n");
    cur_float = params.input_adj;
    for (r = 0; r < params.npl; r++) {
        for (c = 0; c < params.in_h * params.in_w; c++) {
            printf("%*.3lf\t", 6, *cur_float);
            cur_float++;
        }
        puts("");
    }

    printf("\nHidden layers activation:\n");
    cur_uint = params.hidden_act;
    for (r = 0; r < params.hlc; r++) {
        for (c = 0; c < params.npl; c++) {
            printf("%d\t", *cur_uint);
            cur_uint++;
        }
        puts("");
    }

    puts("");
    cur_float = params.hidden_adj;
    for (hl = 0; hl < params.hlc - 1; hl++) {
        printf("Hidden layer %d to %d act:\n", hl + 1, hl + 2);
        for (r = 0; r < params.npl; r++) {
            for (c = 0; c < params.npl; c++) {
                printf("%*.3lf\t", 6, *cur_float);
                cur_float++;
            }
            puts("");
        }
        puts("");
    }

    printf("Hidden layer %d to output act:\n", params.hlc);
    cur_float = params.out_adj;
    for (r = 0; r < BUTTON_COUNT; r++) {
        for (c = 0; c < params.npl; c++) {
            printf("%*.3lf\t", 6, *cur_float);
            cur_float++;
        }
        puts("");
    }

    printf("\nChromosome:\n");
    printf("in_w:\t%d\nin_h:\t%d\nnpl:\t%d\nhlc:\t%d\n", params.in_h, params.in_w, params.npl, params.hlc);
    printf("\nTotal size: %0.2lfKB\n", params.size / 1000.0f);
    printf("-------------------------------------------\n");
}

size_t get_chromosome_size_params(uint8_t in_h, uint8_t in_w, uint8_t hlc, uint16_t npl)
{
    size_t size;

    // Algebra to reduce length of this line
    size = 5 + in_h * (in_w + 4 * in_w * npl) + npl * (4 * npl * (-1 + hlc) + hlc + 4 * BUTTON_COUNT);

    return size;
}

size_t get_chromosome_size(uint8_t *chrom)
{
    size_t size;
    uint8_t in_w, in_h, hlc;
    uint16_t npl;

    in_w = chrom[0];
    in_h = chrom[1];
    npl = *((uint16_t *)chrom + 1);
    hlc = chrom[4];

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
    // if (chance > 0.9f)
    //     return 0.0f;

    // Flip sign on even
    if (rand_r(seedp) % 2 == 0)
        weight = weight * -1;

    return weight;
}

void get_params(uint8_t *chrom, struct params *params)
{
    uint8_t in_w, in_h, hlc;
    uint16_t npl;

    in_w = chrom[0];
    in_h = chrom[1];
    npl = *((uint16_t *)chrom + 1);
    hlc = chrom[4];

    params->in_w = in_w;
    params->in_h = in_h;
    params->npl = npl;
    params->hlc = hlc;

    params->input_act = chrom + 5;
    params->input_adj = (float *)(chrom + 5 + in_h * in_w);
    params->hidden_act = chrom + 5 + in_h * in_w + in_h * in_w * npl * 4;
    params->hidden_adj = (float *)(chrom + 5 + in_h * in_w + in_h * in_w * npl * 4 + npl * hlc);
    params->out_adj = (float *)(chrom + 5 + in_h * in_w + in_h * in_w * npl * 4 + npl * hlc + (hlc - 1) * (npl * npl) * 4);
    params->size = 5 + in_h * (in_w + 4 * in_w * npl) + npl * (4 * npl * (-1 + hlc) + hlc + 4 * BUTTON_COUNT);
}
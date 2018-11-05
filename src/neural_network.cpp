#include <stdio.h>
#include <stdlib.h>
#include <neural_network.hpp>
#include <stdint.h>

uint8_t * generate_chromosome(uint8_t in_h, uint8_t in_w, uint8_t hlc, uint16_t npl);
void print_chromosome(uint8_t *chrom);

int main()
{
    uint8_t *chrom = NULL;
    printf("Running NN with %d by %d inputs, %d outputs, %d hidden layers, and %d nodes per layer.\n", 
        IN_H, IN_W, OUTPUT_SIZE, HLC, NPL);

    chrom = generate_chromosome(IN_H, IN_W, HLC, NPL);
    print_chromosome(chrom);

    free(chrom);
    return 0;
}

uint8_t * generate_chromosome(uint8_t in_h, uint8_t in_w, uint8_t hlc, uint16_t npl)
{
    srand(10);
    size_t total_size;
    uint8_t *chrom = NULL;
    float *tmp;
    int r, c, hl, base;
    
    // Adj matrices multiplied by 4 as they are floats
    total_size = 5 + in_h * in_w + (in_h * in_w) * npl * 4 + hlc * npl + (hlc - 1) * (npl * npl) * 4 + (npl * OUTPUT_SIZE) * 4;

    chrom = (uint8_t *)malloc(total_size);

    chrom[0] = in_w;
    chrom[1] = in_h;
    *(((uint16_t *)chrom) + 1) = npl;
    chrom[4] = hlc;
    
    base = 5;
    // Generate input act matrix
    for (r = 0; r < in_h; r++) {
        for (c = 0; c < in_w; c++) {
            chrom[base] = random() % 2;
            base++;
        }
    }

    // Generate input adj matrix
    tmp = (float *)(chrom + base);
    for (r = 0; r < in_h * in_w; r++) {
        for (c = 0; c < npl; c++) {
            *tmp = (float)random() / (float)random();
            tmp++;
            base += 4;
        }
    }

    // Generate hidden act matrix
    for (r = 0; r < npl; r++) {
        for (c = 0; c < hlc; c++) {
            chrom[base] = random() % 2;
            base++;
        }
    }

    // Generate hidden adj matrices
    tmp = (float *)(chrom + base);
    for (hl = 0; hl < hlc - 1; hl++) {
        for (r = 0; r < npl; r++) {
            for (c = 0; c < npl; c++) {
                *tmp = (float)random() / (float)random();
                tmp++;
                base += 4;
            }
        }
    }

    // Generate out adj matrix
    tmp = (float *)(chrom + base);
    for (r = 0; r < npl; r++) {
        for (c = 0; c < OUTPUT_SIZE; c++) {
            *tmp = (float)random() / (float)random();
            tmp++;
            base += 4;
        }
    }

    return chrom;
}

void print_chromosome(uint8_t *chrom)
{
    printf("-------------------------------------------\n");
    size_t total_size;
    uint8_t in_w, in_h, hlc;
    uint16_t npl;
    float *tmp;
    int r, c, hl, base;

    in_w = chrom[0];
    in_h = chrom[1];
    npl = *((uint16_t *)chrom + 1);
    hlc = chrom[4];

    // Adj matrices multiplied by 4 as they are floats
    total_size = 5 + in_h * in_w + (in_h * in_w) * npl * 4 + hlc * npl + (hlc - 1) * (npl * npl) * 4 + (npl * OUTPUT_SIZE) * 4;

    base = 5;
    // Generate input act matrix
    for (r = 0; r < in_h; r++) {
        for (c = 0; c < in_w; c++) {
            printf("%d\t", chrom[base]);
            base++;
        }
        printf("\n");
    }

    tmp = (float *)(chrom + base);
    for (r = 0; r < in_w * in_h; r++) {
        for (c = 0; c < npl; c++) {
            printf("%0.2lf\t", *tmp);
            tmp++;
            base += 4;
        }
        puts("");
    }

    for (r = 0; r < npl; r++) {
        for (c = 0; c < hlc; c++) {
            printf("%d\t", chrom[base]);
            base++;
        }
        puts("");
    }

    tmp = (float *)(chrom + base);
    for (hl = 0; hl < hlc - 1; hl++) {
        for (r = 0; r < npl; r++) {
            for (c = 0; c < npl; c++) {
                printf("%0.2lf\t", *tmp);
                tmp++;
                base += 4;
            }
            puts("");
        }
        puts("");
    }

    tmp = (float *)(chrom + base);
    for (r = 0; r < npl; r++) {
        for (c = 0; c < OUTPUT_SIZE; c++) {
            printf("%0.2lf\t", *tmp);
            tmp++;
            base += 4;
        }
        puts("");
    }

    printf("\nChromosome:\n");
    printf("in_w:\t%d\nin_h:\t%d\nnpl:\t%d\nhlc:\t%d\n",
              in_h, in_w, npl, hlc);
    printf("\nTotal size: %0.2lfKB\n\n", total_size / 1000.0);
    printf("-------------------------------------------\n");
}
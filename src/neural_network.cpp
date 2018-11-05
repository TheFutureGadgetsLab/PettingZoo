#include <stdio.h>
#include <stdlib.h>
#include <neural_network.hpp>
#include <stdint.h>

uint8_t * generate_chromosome(uint8_t in_h, uint8_t in_w, uint8_t hlc, uint16_t npl);

int main()
{
    uint8_t *chrom = NULL;
    printf("Running NN with %d by %d inputs, %d outputs, %d hidden layers, and %d nodes per layer.\n\n", 
        IN_H, IN_W, OUTPUT_SIZE, HLC, NPL);

    chrom = generate_chromosome(IN_H, IN_W, HLC, NPL);

    free(chrom);
    return 0;
}

uint8_t * generate_chromosome(uint8_t in_h, uint8_t in_w, uint8_t hlc, uint16_t npl)
{
    srand(10);
    size_t total_size, in_act, in_adj, hidden_act, hidden_adj, out_adj;
    uint8_t *chrom = NULL;
    float *tmp;
    int r, c, hl, base;
    
    in_act = in_h * in_w;
    in_adj = (in_h * in_w) * npl * 4;
    hidden_act = hlc * npl;
    hidden_adj = (hlc - 1) * (npl * npl) * 4;
    out_adj = (npl * OUTPUT_SIZE) * 4;

    // Adj matrices multiplied by 4 as they are floats
    total_size = 5 + in_act + in_adj + hidden_act + hidden_adj + out_adj;

    printf("Generating chromosome with the following paramenters:\n");
    printf("  IN_H:\t%d\n  IN_W:\t%d\n  HLC:\t%d\n  NPL:\t%d\n",
              in_h, in_w, hlc, npl);
    printf("\n  Total size: %0.2lfKB\n\n", total_size / 1000.0);
    
    chrom = (uint8_t *)malloc(total_size);

    chrom[0] = in_w;
    chrom[1] = in_h;
    *((uint16_t *)chrom + 1) = npl;
    chrom[4] = HLC;

    printf("IN_W: %d\n", chrom[0]);    
    printf("IN_H: %d\n", chrom[1]);    
    printf("NPL: %d\n", *((uint16_t *)chrom + 1));    
    printf("HLC: %d\n", chrom[4]);    

    base = 5;
    printf("Base: %d\n", base);
    // Generate input act matrix
    for (r = 0; r < in_h; r++) {
        for (c = 0; c < in_w; c++) {
            chrom[base + r * in_w + c] = random() % 2;
            printf("%d\t", chrom[base + r * in_w + c]);
        }
        printf("\n");
    }
    base += in_act;
    
    printf("Base: %d\n", base);
    // Generate input adj matrix
    tmp = (float *)(chrom + base);
    for (r = 0; r < in_h * in_w; r++) {
        for (c = 0; c < npl; c++) {
            *tmp = (float)random() / (float)random();
            printf("%.2lf\t", *tmp);
            tmp++;
        }
        printf("\n");
    }
    base += in_adj;

    printf("Base: %d\n", base);
    // Generate hidden act matrix
    for (r = 0; r < npl; r++) {
        for (c = 0; c < hlc; c++) {
            chrom[base + r * hlc + c] = random() % 2;
            printf("%d\t", chrom[base + r * hlc + c]);
        }
        printf("\n");
    }
    base += hidden_act;

    // Generate hidden adj matrices
    printf("Base: %d\n", base);
    tmp = (float *)(chrom + base);
    for (hl = 0; hl < hlc - 1; hl++) {
        for (r = 0; r < in_h * in_w; r++) {
            for (c = 0; c < npl; c++) {
                *tmp = (float)random() / (float)random();
                printf("%.2lf\t", *tmp);
                tmp++;
            }
            puts("");
        }
        puts("");
    }
    base += hidden_adj;
    
    printf("Base: %d\n", base);
    // Generate out adj matrix
    tmp = (float *)(chrom + base);
    for (r = 0; r < npl; r++) {
        for (c = 0; c < OUTPUT_SIZE; c++) {
            *tmp = (float)random() / (float)random();
            printf("%.2lf\t", *tmp);
            tmp++;
        }
        printf("\n");
    }

    base += out_adj;
    printf("Base: %d\n", base);

    return chrom;
}
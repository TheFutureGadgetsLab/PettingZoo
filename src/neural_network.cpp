#include <stdio.h>
#include <stdlib.h>
#include <chromosome.hpp>
#include <neural_network.hpp>
#include <stdint.h>
#include <time.h>
#include <math.h>

void calc_first_layer(uint8_t *chrom, uint8_t *inputs, float *node_outputs);

// Activation functions
float sigmoid(float x);
float softsign(float x);
float sigmoid_bounded(float x);
float softsign_bounded(float x);
float tanh_bounded(float x);

int main()
{
    uint8_t *chrom = NULL;
    float *node_outputs = NULL;
    float *network_outputs = NULL;
    uint8_t inputs[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};

    printf("Running NN with %d by %d inputs, %d outputs, %d hidden layers, and %d nodes per layer.\n", 
        IN_H, IN_W, BUTTON_COUNT, HLC, NPL);

    // srand(time(NULL));
    srand(10);

    chrom = generate_chromosome(IN_H, IN_W, HLC, NPL);
    print_chromosome(chrom);

    node_outputs = (float *)malloc(sizeof(float) * NPL * HLC);
    network_outputs = (float *)malloc(sizeof(float) * BUTTON_COUNT);

    puts("Inputs: ");
    for(int row = 0; row < IN_H; row++) {
        for(int col = 0; col < IN_W; col++) {
            printf("%d\t", inputs[row * IN_W + col]);
        }
        puts("");
    }

    calc_first_layer(chrom, inputs, node_outputs);

    printf("node_outputs:\n");
    for(int row = 0; row < NPL; row++) {
        printf("%lf\t%lf\n",
            node_outputs[row * HLC + 0],
            node_outputs[row * HLC + 1]);
    }

    free(chrom);

    return 0;
}

void calc_first_layer(uint8_t *chrom, uint8_t *inputs, float *node_outputs)
{
    int node, weight;
    float sum;
    float *input_adj;
    uint8_t *input_act;
    uint8_t in_w, in_h, hlc, input;
    uint16_t npl;

    in_w = chrom[0];
    in_h = chrom[1];
    npl = *((uint16_t *)chrom + 1);
    hlc = chrom[4];

    input_adj = locate_input_adj(chrom);
    input_act = locate_input_act(chrom);

    for (node = 0; node < npl; node++) {
        sum = 0.0f;
        printf("HL 0, node %d\n", node);
        for (weight = 0; weight < in_h * in_w; weight++) {
            sum += input_adj[node * in_h * in_w + weight] * inputs[weight];
            printf("%0.3f * %d\n", input_adj[node * in_h * in_w + weight], inputs[weight]);
        }
        // node_outputs[node] = sigmoid(sum);
        node_outputs[node * hlc] = sum;
        puts("");
    }
}

void calc_hidden_layers(uint8_t *chrom, uint8_t *inputs, float *node_outputs)
{
    int node, weight, layer;
    float sum;
    float *hidden_adj;
    uint8_t in_w, in_h, hlc, input;
    uint16_t npl;

    in_w = chrom[0];
    in_h = chrom[1];
    npl = *((uint16_t *)chrom + 1);
    hlc = chrom[4];

    // input_adj = locate_hidden_adj(chrom);

    for (layer = 0; layer < hlc; layer++) {

    }
    for (node = 0; node < npl; node++) {
        sum = 0.0f;
        printf("Node %d\n", node);
        for (weight = 0; weight < in_h * in_w; weight++) {
            // sum += input_adj[weight * npl + node] * inputs[weight];
        }
        // node_outputs[node] = sigmoid(sum);
        node_outputs[node] = sum;
    }
}

// sigmoid in [-1,1]
float sigmoid(float x)
{
    return 2.0 / (1.0 + expf(-x)) - 1.0;
}

// x/(1+|x|) in [-1,1]
float softsign(float x)
{
    return x / (1.0 + abs(x));
}

// sigmoid in [0,1]
float sigmoid_bounded(float x)
{
    return 1.0 / (1.0 + expf(-x));
}

// x/(1+|x|) in [0,1]
float softsign_bounded(float x)
{
    return (0.5 * x) / (1.0 + abs(x)) + 0.5;
}

// tanh(x) in [0,1]
float tanh_bounded(float x)
{
    return 0.5 + tanh(x) * 0.5;
}

// float vec_dot(float *a, float *b, int size)
// {

//     return 0.0f;
// }
#include <stdio.h>
#include <stdlib.h>
#include <chromosome.hpp>
#include <neural_network.hpp>
#include <stdint.h>
#include <time.h>
#include <math.h>

void calc_first_layer(uint8_t *chrom, uint8_t *inputs, float *node_outputs);
void calc_hidden_layers(uint8_t *chrom, float *node_outputs);
void calc_output(uint8_t *chrom, float *node_outputs, float *network_outputs);

int main()
{
    uint8_t *chrom = NULL;
    float *node_outputs = NULL;
    float *network_outputs = NULL;
    uint8_t *inputs = (uint8_t *)malloc(sizeof(uint8_t) * IN_W * IN_H);

    // Initialize example input array
    for(int row = 0; row < IN_H; row++) {
        for(int col = 0; col < IN_W; col++) {
            inputs[row * IN_W + col] = row * IN_W + col;
        }
    }

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
    puts("");

    calc_first_layer(chrom, inputs, node_outputs);
    calc_hidden_layers(chrom, node_outputs);
    calc_output(chrom, node_outputs, network_outputs);

    printf("node_outputs:\n");
    for(int row = 0; row < HLC; row++) {
        for (int col = 0; col < NPL; col++) {
            printf("%*.4lf\t", 6, node_outputs[row * NPL + col]);
        }
        puts("");
    }
    puts("");

    printf("network_outputs:\n");
    for(int button = 0; button < BUTTON_COUNT; button++) {
        printf("%*.4lf\t", 6, network_outputs[button]);
    }
    puts("");

    free(chrom);

    return 0;
}

void calc_first_layer(uint8_t *chrom, uint8_t *inputs, float *node_outputs)
{
    int node, weight;
    float sum;
    float *input_adj;
    uint8_t *input_act, *hidden_act;
    uint8_t input;
    struct params params;

    get_params(chrom, &params);

    input_adj = locate_input_adj(chrom);
    input_act = locate_input_act(chrom);
    hidden_act = locate_hidden_act(chrom);

    for (node = 0; node < params.npl; node++) {
        sum = 0.0f;

        printf("HL 0, node %d, active: %d\n", node, hidden_act[node]);
        if (!hidden_act[node]) {
            printf("Inactive node, skipping\n\n");
            node_outputs[node] = 0.0f;
            continue;
        }

        for (weight = 0; weight < params.in_h * params.in_w; weight++) {
            // Branchless input. If input tile is inactive, the weight
            // gets multiplied by 0, otherwise by 1.
            input = inputs[weight] * input_act[weight];
            sum += input_adj[node * params.in_h * params.in_w + weight] * input;
            printf("%*.3f * %d\n", 6, input_adj[node * params.in_h * params.in_w + weight], input);
        }

        node_outputs[node] = sigmoid(sum);
        printf("----------\nSum: %.3lf\nAct: %.3lf\n\n", sum, node_outputs[node]);
    }
}

void calc_hidden_layers(uint8_t *chrom, float *node_outputs)
{
    int node, weight, layer;
    float sum, input;
    float *hidden_adj;
    uint8_t *hidden_act;
    struct params params;

    get_params(chrom, &params);

    hidden_act = locate_hidden_act(chrom);

    for (layer = 1; layer < params.hlc; layer++) {
        hidden_adj = locate_hidden_adj(chrom, layer - 1);
        for (node = 0; node < params.npl; node++) {
            sum = 0.0f;
            printf("HL %d, node %d, active: %d\n", layer, node, hidden_act[layer * params.npl + node]);
            if (!hidden_act[layer * params.npl + node]) {
                printf("Inactive node, skipping\n\n");
                node_outputs[layer * params.npl + node] = 0.0f;
                continue;
            }

            for (weight = 0; weight < params.npl; weight++) {
                input = node_outputs[(layer - 1) * params.npl + weight];
                sum += hidden_adj[node * params.npl + weight] * input;
                printf("%*.3f * %*.3lf\n", 6, hidden_adj[node * params.npl + weight], 6, input);
            }

            node_outputs[layer * params.npl + node] = sigmoid(sum);
            printf("----------\nSum: %.3lf\nAct: %.3lf\n\n", sum, node_outputs[layer * params.npl + node]);
        }
    }
}

void calc_output(uint8_t *chrom, float *node_outputs, float *network_outputs)
{
    int out, weight;
    float sum;
    float *out_adj;
    float input;
    struct params params;

    get_params(chrom, &params);

    out_adj = locate_out_adj(chrom);

    for (out = 0; out < BUTTON_COUNT; out++) {
        sum = 0.0f;
        printf("Output, node %d\n", out);
        for (weight = 0; weight < params.npl; weight++) {
            input = node_outputs[(params.hlc - 1) * params.npl + weight];
            sum += out_adj[out * params.npl + weight] * input;
            printf("%*.3f * %*.3lf\n", 6, out_adj[out * params.npl + weight], 6, input);
        }

        network_outputs[out] = sigmoid(sum);
        printf("----------\nSum: %.3lf\nAct: %.3lf\n\n", sum, network_outputs[out]);
    }
}

float sigmoid(float x)
{
    return 1.0f / (1 + expf(-x));
}

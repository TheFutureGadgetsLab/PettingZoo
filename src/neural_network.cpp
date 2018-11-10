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
    uint8_t *inputs = NULL;
    int row, col, button;

    inputs = (uint8_t *)malloc(sizeof(uint8_t) * IN_W * IN_H);
    node_outputs = (float *)malloc(sizeof(float) * NPL * HLC);
    network_outputs = (float *)malloc(sizeof(float) * BUTTON_COUNT);

    // Initialize example input array
    for (row = 0; row < IN_H; row++) {
        for (col = 0; col < IN_W; col++) {
            inputs[row * IN_W + col] = row * IN_W + col;
        }
    }

    srand(10);

    chrom = generate_chromosome(IN_H, IN_W, HLC, NPL);

    calc_first_layer(chrom, inputs, node_outputs);
    calc_hidden_layers(chrom, node_outputs);
    calc_output(chrom, node_outputs, network_outputs);

    printf("node_outputs:\n");
    for (row = 0; row < HLC; row++) {
        for (col = 0; col < NPL; col++) {
            printf("%lf\t", node_outputs[row * NPL + col]);
        }
        puts("");
    }

    printf("\nnetwork_outputs:\n");
    for (button = 0; button < BUTTON_COUNT; button++) {
        printf("%lf\t", network_outputs[button]);
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
    struct params prms;

    get_params(chrom, &prms);

    input_adj = locate_input_adj(chrom);
    input_act = locate_input_act(chrom);
    hidden_act = locate_hidden_act(chrom);

    // Loop over nodes in the first hidden layer
    for (node = 0; node < prms.npl; node++) {
        sum = 0.0f;

        // If the node isn't active its output defaults to 0
        if (!hidden_act[node]) {
            node_outputs[node] = 0.0f;
            continue;
        }

        // Calculate linear sum of outputs and weights
        for (weight = 0; weight < prms.in_h * prms.in_w; weight++) {
            // Branchless input. If input tile is inactive, the weight
            // gets multiplied by 0, otherwise by 1.
            sum += input_adj[node * prms.in_h * prms.in_w + weight] * 
                   inputs[weight] * input_act[weight];
        }

        node_outputs[node] = sigmoid_bounded(sum);
    }
}

void calc_hidden_layers(uint8_t *chrom, float *node_outs)
{
    int node, weight, layer, cur_node;
    float sum;
    float *hidden_adj;
    uint8_t *hidden_act;
    struct params prms;

    get_params(chrom, &prms);

    hidden_act = locate_hidden_act(chrom);

    // Loop over layers, beginning at 2nd (first is handled differently)
    for (layer = 1; layer < prms.hlc; layer++) {
        // Grab the adjacency matrix for this layer
        hidden_adj = locate_hidden_adj(chrom, layer - 1);
        // Loop over nodes in this layer
        for (node = 0; node < prms.npl; node++) {
            sum = 0.0f;
            cur_node = layer * prms.npl + node;

            // If the node isn't active its output defaults to 0
            if (!hidden_act[cur_node]) {
                node_outs[cur_node] = 0.0f;
                continue;
            }

            // Calculate linear sum of outputs and weights
            for (weight = 0; weight < prms.npl; weight++) {
                sum += hidden_adj[node * prms.npl + weight] * 
                       node_outs[(layer - 1) * prms.npl + weight];
            }

            node_outs[cur_node] = sigmoid_bounded(sum);
        }
    }
}

void calc_output(uint8_t *chrom, float *node_outs, float *net_outs)
{
    int bttn, weight;
    float sum;
    float *out_adj;
    struct params prms;

    get_params(chrom, &prms);

    out_adj = locate_out_adj(chrom);

    // Loop over buttons
    for (bttn = 0; bttn < BUTTON_COUNT; bttn++) {
        sum = 0.0f;
        // Linear sum
        for (weight = 0; weight < prms.npl; weight++) {
            sum += out_adj[bttn * prms.npl + weight] * 
                   node_outs[(prms.hlc - 1) * prms.npl + weight];
        }

        net_outs[bttn] = sigmoid_bounded(sum);
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
#include <stdio.h>
#include <stdlib.h>
#include <chromosome.hpp>
#include <neural_network.hpp>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <gamelogic.hpp>
#include <sys/stat.h>

__host__ __device__
void calc_first_layer(struct chromosome *chrom, float *inputs, float *node_outputs);
__host__ __device__
void calc_hidden_layers(struct chromosome *chrom, float *node_outputs);
__host__ __device__
void calc_output(struct chromosome *chrom, float *node_outputs, float *network_outputs);

// Activation functions
__host__ __device__
float sigmoid(float x);
__host__ __device__
float softsign(float x);
__host__ __device__
float sigmoid_bounded(float x);
__host__ __device__
float softsign_bounded(float x);
__host__ __device__
float tanh_bounded(float x);

__host__ __device__
int evaluate_frame(struct Game *game, struct Player *player, struct chromosome *chrom, float *tiles, float *node_outputs, uint8_t *buttons)
{
    float network_outputs[BUTTON_COUNT];
    uint8_t inputs[BUTTON_COUNT];
    int ret;

    get_input_tiles(game, player, tiles, chrom->in_h, chrom->in_w);
    
    calc_first_layer(chrom, tiles, node_outputs);
    calc_hidden_layers(chrom, node_outputs);
    calc_output(chrom, node_outputs, network_outputs);

    // Assign button presses based on output probability
    inputs[BUTTON_RIGHT] = network_outputs[BUTTON_RIGHT] > 0.0f;
    inputs[BUTTON_LEFT] = network_outputs[BUTTON_LEFT] > 0.0f;
    inputs[BUTTON_JUMP] = network_outputs[BUTTON_JUMP] > 0.0f;

    // Add pressed buttons to the buffer
    *buttons = inputs[BUTTON_RIGHT]       |
               (inputs[BUTTON_LEFT] << 1) | 
               (inputs[BUTTON_JUMP] << 2);

    ret = game_update(game, player, inputs);

    return ret;
}

__host__ __device__
void calc_first_layer(struct chromosome *chrom, float *inputs, float *node_outputs)
{
    int node, weight;
    float sum;

    // Loop over nodes in the first hidden layer
    for (node = 0; node < chrom->npl; node++) {
        sum = 0.0f;

        // If the node isn't active its output defaults to 0
        if (!chrom->hidden_act[node]) {
            node_outputs[node] = 0.0f;
            continue;
        }

        // Calculate linear sum of outputs and weights
        for (weight = 0; weight < chrom->in_h * chrom->in_w; weight++) {
            if (chrom->input_act[weight])
                sum += chrom->input_adj[node * chrom->in_h * chrom->in_w + weight] * inputs[weight];
        }

        node_outputs[node] = softsign(sum);
    }
}

__host__ __device__
void calc_hidden_layers(struct chromosome *chrom, float *node_outs)
{
    int node, weight, layer, cur_node;
    float sum;
    float *hidden_adj;

    // Loop over layers, beginning at 2nd (first is handled by calc_first_layer)
    for (layer = 1; layer < chrom->hlc; layer++) {
        // Grab the adjacency matrix for this layer
        hidden_adj = chrom->hidden_adj + (layer - 1) * chrom->npl * chrom->npl;
        // Loop over nodes in this layer
        for (node = 0; node < chrom->npl; node++) {
            sum = 0.0f;
            cur_node = layer * chrom->npl + node;

            // If the node isn't active its output defaults to 0
            if (!chrom->hidden_act[cur_node]) {
                node_outs[cur_node] = 0.0f;
                continue;
            }

            // Calculate linear sum of outputs and weights
            for (weight = 0; weight < chrom->npl; weight++) {
                sum += hidden_adj[node * chrom->npl + weight] * node_outs[(layer - 1) * chrom->npl + weight];
            }

            node_outs[cur_node] = softsign(sum);
        }
    }
}

__host__ __device__
void calc_output(struct chromosome *chrom, float *node_outs, float *net_outs)
{
    int bttn, weight;
    float sum;

    // Loop over buttons
    for (bttn = 0; bttn < BUTTON_COUNT; bttn++) {
        sum = 0.0f;
        // Linear sum
        for (weight = 0; weight < chrom->npl; weight++) {
            sum += chrom->out_adj[bttn * chrom->npl + weight] * node_outs[(chrom->hlc - 1) * chrom->npl + weight];
        }

        net_outs[bttn] = softsign(sum);
    }
}

// sigmoid in (-1,1)
__host__ __device__
float sigmoid(float x)
{
    return 2.0f / (1.0f + expf(-x)) - 1.0;
}

// x/(1+|x|) in [-1,1]
__host__ __device__
float softsign(float x)
{
    return x / (1.0f + fabs(x));
}

// sigmoid in (0,1)
__host__ __device__
float sigmoid_bounded(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

// x/(1+|x|) in [0,1]
__host__ __device__
float softsign_bounded(float x)
{
    return (0.5f * x) / (1.0f + fabs(x)) + 0.5;
}

// tanh(x) in (0,1)
__host__ __device__
float tanh_bounded(float x)
{
    return 0.5f + tanhf(x) * 0.5f;
}
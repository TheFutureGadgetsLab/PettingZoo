/**
 * @file neural_network.cpp
 * @author Haydn Jones, Benjamin Mastripolito
 * @brief Routines for running a neural network
 * @date 2018-12-06
 */

#include <stdio.h>
#include <stdlib.h>
#include <chromosome.hpp>
#include <neural_network.hpp>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <gamelogic.hpp>
#include <cuda_runtime.h>

__host__ __device__
void calc_first_layer(struct Chromosome *chrom, float *inputs, float *node_outputs);
__host__ __device__
void calc_hidden_layers(struct Chromosome *chrom, float *node_outputs);
__host__ __device__
void calc_output(struct Chromosome *chrom, float *node_outputs, float *network_outputs);

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

/**
 * @brief Evaluates a single frame of the game with the given chromosome
 * 
 * @param game The game struct
 * @param player The player struct that this chromosome will be controlling
 * @param chrom The chromosome
 * @param buttons The button presses the chromosome outputs will be placed here
 * @param tiles The game tiles
 * @param node_outputs The node outputs
 * @return int PLAYER_DEAD or PLAYER_COMPLETE
 */
__host__ __device__
int evaluate_frame(struct Game *game, struct Player *player, struct Chromosome *chrom, uint8_t *buttons, float *tiles, float *node_outputs)
{
    float network_outputs[BUTTON_COUNT];
    uint8_t inputs[BUTTON_COUNT];

    get_input_tiles(game, player, tiles, chrom->in_h, chrom->in_w);
    
    calc_first_layer(chrom, tiles, node_outputs);
    calc_hidden_layers(chrom, node_outputs);
    calc_output(chrom, node_outputs, network_outputs);

    // Assign button presses based on output probability
    inputs[BUTTON_RIGHT] = network_outputs[BUTTON_RIGHT] > 0.0f;
    inputs[BUTTON_LEFT] = network_outputs[BUTTON_LEFT] > 0.0f;
    inputs[BUTTON_JUMP] = network_outputs[BUTTON_JUMP] > 0.0f;

    *buttons = inputs[BUTTON_RIGHT] | (inputs[BUTTON_LEFT] << 1) | (inputs[BUTTON_JUMP] << 2);

    return game_update(game, player, inputs);
}

/**
 * @brief Calculates output of first hidden layer
 * 
 * @param chrom Chromosome to use
 * @param inputs Input tiles for network
 * @param node_outputs Where to store node outputs
 */
__host__ __device__
void calc_first_layer(struct Chromosome *chrom, float *inputs, float *node_outputs)
{
    int node, weight;
    float sum;

    // Loop over nodes in the first hidden layer
    for (node = 0; node < chrom->npl; node++) {
        sum = 0.0f;

        // Calculate linear sum of outputs and weights
        for (weight = 0; weight < chrom->in_h * chrom->in_w; weight++) {
            sum += chrom->input_adj[node * chrom->in_h * chrom->in_w + weight] * inputs[weight];
        }

        node_outputs[node] = softsign(sum);
    }
}

/**
 * @brief Calculate the chromosome's hidden layers
 * 
 * @param chrom The chromosome being simulated
 * @param node_outs Outputs for the nodes
 */
__host__ __device__
void calc_hidden_layers(struct Chromosome *chrom, float *node_outs)
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

            // Calculate linear sum of outputs and weights
            for (weight = 0; weight < chrom->npl; weight++) {
                sum += hidden_adj[node * chrom->npl + weight] * node_outs[(layer - 1) * chrom->npl + weight];
            }

            node_outs[cur_node] = softsign(sum);
        }
    }
}

/**
 * @brief Calculates the outputs of the network
 * 
 * @param chrom Chromosome to use
 * @param node_outs Outputs from previous layer
 * @param net_outs Where to store network outputs
 */
__host__ __device__
void calc_output(struct Chromosome *chrom, float *node_outs, float *net_outs)
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

/**
 * @brief The sigmoid function, bounded between -1 and 1
 * 
 * @param x Input
 * @return float Sigmoid output
 */
__host__ __device__
float sigmoid(float x)
{
    return 2.0f / (1.0f + expf(-x)) - 1.0;
}

/**
 * @brief The softsign function, bounded between -1 and 1
 * 
 * @param x The input value
 * @return float Output value
 */
__host__ __device__
float softsign(float x)
{
    return x / (1.0f + fabs(x));
}

/**
 * @brief sigmoid in (0,1)
 * 
 * @param x Input to function
 * @return float Output
 */
__host__ __device__
float sigmoid_bounded(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

/**
 * @brief The softsign function
 * 
 * @param x 
 * @return float 
 */
__host__ __device__
float softsign_bounded(float x)
{
    return (0.5f * x) / (1.0f + fabs(x)) + 0.5;
}

/**
 * @brief tanh(x) in (0,1)
 * 
 * @param x Input to function
 * @return float Output
 */
__host__ __device__
float tanh_bounded(float x)
{
    return 0.5f + tanhf(x) * 0.5f;
}
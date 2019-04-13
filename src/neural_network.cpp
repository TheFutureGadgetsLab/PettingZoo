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
#include <time.h>
#include <math.h>
#include <gamelogic.hpp>

void calc_first_layer(Chromosome& chrom, std::vector<float>& inputs, std::vector<float>& node_outputs);
void calc_hidden_layers(Chromosome& chrom, std::vector<float>& node_outputs);
void calc_output(Chromosome& chrom, std::vector<float>& node_outputs, float* net_outs);

// Activation functions
float sigmoid(float x);
float softsign(float x);
float sigmoid_bounded(float x);
float softsign_bounded(float x);
float tanh_bounded(float x);

/**
 * @brief Evaluates a single frame of the game with the given chromosome
 * 
 * @param game The game obj
 * @param player The player that this chromosome will be controlling
 * @param chrom The chromosome
 * @param buttons The button presses the chromosome outputs will be placed here
 * @param tiles The game tiles
 * @param node_outputs The node outputs
 * @return int PLAYER_DEAD or PLAYER_COMPLETE
 */
int evaluate_frame(Game& game, Player& player, Chromosome& chrom)
{
    float network_outputs[BUTTON_COUNT];

    game.getInputTiles(player, chrom.input_tiles, chrom.in_h, chrom.in_w);
    
    calc_first_layer(chrom, chrom.input_tiles, chrom.node_outputs);
    calc_hidden_layers(chrom, chrom.node_outputs);
    calc_output(chrom, chrom.node_outputs, network_outputs);

    // Assign button presses based on output probability
    player.right = network_outputs[RIGHT] > 0.0f;
    player.left = network_outputs[LEFT] > 0.0f;
    player.jump = network_outputs[JUMP] > 0.0f;

    return 0;
}

/**
 * @brief Calculates output of first hidden layer
 * 
 * @param chrom Chromosome to use
 * @param inputs Input tiles for network
 * @param node_outputs Where to store node outputs
 */
void calc_first_layer(Chromosome& chrom, std::vector<float>& inputs, std::vector<float>& node_outputs)
{
    int node, weight;
    float sum;

    // Loop over nodes in the first hidden layer
    for (node = 0; node < chrom.npl; node++) {
        sum = 0.0f;

        // Calculate linear sum of outputs and weights
        for (weight = 0; weight < chrom.in_h * chrom.in_w; weight++) {
            sum += chrom.input_adj[node * chrom.in_h * chrom.in_w + weight] * inputs[weight];
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
void calc_hidden_layers(Chromosome& chrom, std::vector<float>& node_outs)
{
    int node, weight, layer, cur_node;
    float sum;
    float *hidden_adj;

    // Loop over layers, beginning at 2nd (first is handled by calc_first_layer)
    for (int layer = 0; layer < chrom.hiddenLayers.size(); layer++) {
        for (node = 0; node < chrom.npl; node++) {
            sum = 0.0f;
            cur_node = (layer + 1) * chrom.npl + node;

            // Calculate linear sum of outputs and weights
            for (weight = 0; weight < chrom.npl; weight++) {
                sum += chrom.hiddenLayers[layer][node * chrom.npl + weight] * node_outs[layer * chrom.npl + weight];
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
void calc_output(Chromosome& chrom, std::vector<float>& node_outs, float* net_outs)
{
    int bttn, weight;
    float sum;

    // Loop over buttons
    for (bttn = 0; bttn < BUTTON_COUNT; bttn++) {
        sum = 0.0f;
        // Linear sum
        for (weight = 0; weight < chrom.npl; weight++) {
            sum += chrom.out_adj[bttn * chrom.npl + weight] * node_outs[(chrom.hlc - 1) * chrom.npl + weight];
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
float tanh_bounded(float x)
{
    return 0.5f + tanhf(x) * 0.5f;
}
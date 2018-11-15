#include <stdio.h>
#include <stdlib.h>
#include <chromosome.hpp>
#include <neural_network.hpp>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <gamelogic.hpp>
#include <sys/stat.h>

void calc_first_layer(uint8_t *chrom, uint8_t *inputs, float *node_outputs);
void calc_hidden_layers(uint8_t *chrom, float *node_outputs);
void calc_output(uint8_t *chrom, float *node_outputs, float *network_outputs);
int evaluate_frame(struct Game *game, struct Player *player, uint8_t *chrom, uint8_t *tiles, float *node_outputs, uint8_t *buttons);
void write_out(uint8_t *buttons, size_t buttons_bytes, uint8_t *chrom, unsigned int seed);

// Activation functions
float sigmoid(float x);
float softsign(float x);
float sigmoid_bounded(float x);
float softsign_bounded(float x);
float tanh_bounded(float x);

int main()
{
    uint8_t *chrom = NULL;
    uint8_t *tiles = NULL;
    float *node_outputs = NULL;
    struct Game game;
    struct Player player;
    uint8_t buttons[MAX_FRAMES];
    int buttons_index, ret;
    unsigned int seed;
    char resp;

    tiles = (uint8_t *)malloc(sizeof(uint8_t) * IN_W * IN_H);
    node_outputs = (float *)malloc(sizeof(float) * NPL * HLC);
    chrom = (uint8_t *)malloc(sizeof(uint8_t) * get_chromosome_size_params(IN_H, IN_W, HLC, NPL));
    
    seed = time(NULL);
    printf("Seed = %u\n", seed);

    game_setup(&game, &player, seed);
    generate_chromosome(chrom, IN_H, IN_W, HLC, NPL, seed);
    
    buttons_index = 0;
    while (1) {
        ret = evaluate_frame(&game, &player, chrom, tiles, node_outputs, buttons + buttons_index);
        buttons_index++;

        if (ret == PLAYER_DEAD || ret == PLAYER_TIMEOUT)
            break;
    }
    
    if (ret == PLAYER_DEAD)
        printf("Player died\n");
    else
        printf("Player timed out\n");
    printf("Fitness: %d\n", player.fitness);

    printf("Would you like to write this out (y/n)?: ");
    resp = getchar();
    if (resp == 'y')
        write_out(buttons, MAX_FRAMES, chrom, seed);

    free(chrom);
    free(tiles);
    free(node_outputs);

    return 0;
}

int evaluate_frame(struct Game *game, struct Player *player, uint8_t *chrom, uint8_t *tiles, float *node_outputs, uint8_t *buttons)
{
    struct params prms;
    float network_outputs[BUTTON_COUNT];
    uint8_t inputs[BUTTON_COUNT];
    int ret;

    get_params(chrom, &prms);
    get_input_tiles(game, player, tiles, prms.in_h, prms.in_w);
    
    calc_first_layer(chrom, tiles, node_outputs);
    calc_hidden_layers(chrom, node_outputs);
    calc_output(chrom, node_outputs, network_outputs);

    // Assign button presses based on output probability
    inputs[BUTTON_RIGHT] = network_outputs[BUTTON_RIGHT] > 0.5f;
    inputs[BUTTON_LEFT] = network_outputs[BUTTON_LEFT] > 0.5f;
    inputs[BUTTON_JUMP] = network_outputs[BUTTON_JUMP] > 0.5f;

    // Add pressed buttons to the buffer
    *buttons = inputs[BUTTON_RIGHT]       |
               (inputs[BUTTON_LEFT] << 1) | 
               (inputs[BUTTON_JUMP] << 2);

    ret = game_update(game, player, inputs);

    return ret;
}

void calc_first_layer(uint8_t *chrom, uint8_t *inputs, float *node_outputs)
{
    int node, weight;
    float sum;
    struct params prms;

    get_params(chrom, &prms);

    // Loop over nodes in the first hidden layer
    for (node = 0; node < prms.npl; node++) {
        sum = 0.0f;

        // If the node isn't active its output defaults to 0
        if (!prms.hidden_act[node]) {
            node_outputs[node] = 0.0f;
            continue;
        }

        // Calculate linear sum of outputs and weights
        for (weight = 0; weight < prms.in_h * prms.in_w; weight++) {
            // Branchless input. If input tile is inactive, the weight
            // gets multiplied by 0, otherwise by 1.
            sum += prms.input_adj[node * prms.in_h * prms.in_w + weight] * 
                   inputs[weight] * prms.input_act[weight];
        }

        node_outputs[node] = sigmoid_bounded(sum);
    }
}

void calc_hidden_layers(uint8_t *chrom, float *node_outs)
{
    int node, weight, layer, cur_node;
    float sum;
    float *hidden_adj;
    struct params prms;

    get_params(chrom, &prms);

    // Loop over layers, beginning at 2nd (first is handled by calc_first_layer)
    for (layer = 1; layer < prms.hlc; layer++) {
        // Grab the adjacency matrix for this layer
        // hidden_adj = locate_hidden_adj(chrom, layer - 1);
        hidden_adj = prms.hidden_adj + (layer - 1) * prms.npl * prms.npl;
        // Loop over nodes in this layer
        for (node = 0; node < prms.npl; node++) {
            sum = 0.0f;
            cur_node = layer * prms.npl + node;

            // If the node isn't active its output defaults to 0
            if (!prms.hidden_act[cur_node]) {
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
    struct params prms;

    get_params(chrom, &prms);

    // Loop over buttons
    for (bttn = 0; bttn < BUTTON_COUNT; bttn++) {
        sum = 0.0f;
        // Linear sum
        for (weight = 0; weight < prms.npl; weight++) {
            sum += prms.out_adj[bttn * prms.npl + weight] * 
                   node_outs[(prms.hlc - 1) * prms.npl + weight];
        }

        net_outs[bttn] = sigmoid_bounded(sum);
    }
}

void write_out(uint8_t *buttons, size_t buttons_bytes, uint8_t *chrom, unsigned int seed)
{
    char fname[128];
    sprintf(fname, "%d.bin", seed);
    FILE *file = fopen(fname, "wb");

    size_t chrom_bytes = get_chromosome_size(chrom);

    //Seed
    fwrite(&seed, sizeof(unsigned int), 1, file);

    //Button presses
    fwrite(buttons, sizeof(uint8_t), buttons_bytes, file);

    //Chromosome
    fwrite(chrom, sizeof(uint8_t), chrom_bytes, file);

    fclose(file);
}

// sigmoid in (-1,1)
float sigmoid(float x)
{
    return 2.0 / (1.0 + expf(-x)) - 1.0;
}

// x/(1+|x|) in [-1,1]
float softsign(float x)
{
    return x / (1.0 + abs(x));
}

// sigmoid in (0,1)
float sigmoid_bounded(float x)
{
    return 1.0 / (1.0 + expf(-x));
}

// x/(1+|x|) in [0,1]
float softsign_bounded(float x)
{
    return (0.5 * x) / (1.0 + abs(x)) + 0.5;
}

// tanh(x) in (0,1)
float tanh_bounded(float x)
{
    return 0.5 + tanh(x) * 0.5;
}
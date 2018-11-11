#include <stdio.h>
#include <stdlib.h>
#include <chromosome.hpp>
#include <neural_network.hpp>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <gamelogic.hpp>

void calc_first_layer(uint8_t *chrom, uint8_t *inputs, float *node_outputs);
void calc_hidden_layers(uint8_t *chrom, float *node_outputs);
void calc_output(uint8_t *chrom, float *node_outputs, float *network_outputs);
int evaluate_frame(struct Game *game, struct Player *player, uint8_t *chrom, uint8_t *tiles, float *node_outputs, uint8_t *buttons);
void write_out(uint8_t *buttons, size_t buttons_bytes, uint8_t *chrom, size_t chrom_bytes, unsigned int seed);

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
    uint8_t *tiles = NULL;
    struct Game game;
    struct Player player;
    uint8_t *buttons = NULL;
    uint buttons_index = 0;
    unsigned seed = time(NULL);

    tiles = (uint8_t *)malloc(sizeof(uint8_t) * IN_W * IN_H);
    node_outputs = (float *)malloc(sizeof(float) * NPL * HLC);
    buttons = (uint8_t *)malloc(sizeof(uint8_t) * MAX_FRAMES);

    uint seed = time(NULL);
    srand(seed);

    chrom = generate_chromosome(IN_H, IN_W, HLC, NPL);

    game_setup(&game, &player, seed);
    while (evaluate_frame(&game, &player, chrom, tiles, node_outputs, buttons) != -1) {
        game.frame += 1;
        continue;
    }    

    printf("PLAYER DEAD\n SCORE: %d\n FITNESS: %d\n", player.score, player.fitness);

    struct params prms;
    get_params(chrom, &prms);
    write_out(buttons, MAX_FRAMES, chrom, get_chromosome_size(prms), seed);

    free(chrom);
    free(tiles);
    free(node_outputs);
    free(buttons);

    return 0;
}

int evaluate_frame(struct Game *game, struct Player *player, uint8_t *chrom, uint8_t *tiles, float *node_outputs, uint8_t *buttons)
{
    struct params prms;
    float network_outputs[BUTTON_COUNT];
    int inputs[BUTTON_COUNT];
    int x, y;

    get_params(chrom, &prms);
    get_input_tiles(game, player, tiles, prms.in_h, prms.in_w);

    
    /* for(y = 0; y < prms.in_h; y++) {
        for(x = 0; x < prms.in_w; x++) {
            if (tiles[y * prms.in_w + x] > 0) {
                printf("0 ");
            } else {
                printf("  ");
            }
        }
        puts("");
    } */
    
    calc_first_layer(chrom, tiles, node_outputs);
    calc_hidden_layers(chrom, node_outputs);
    calc_output(chrom, node_outputs, network_outputs);

    // Assign button presses based on output probability
    inputs[BUTTON_RIGHT] = network_outputs[BUTTON_RIGHT] > 0.5f;
    inputs[BUTTON_LEFT] = network_outputs[BUTTON_LEFT] > 0.5f;
    inputs[BUTTON_JUMP] = network_outputs[BUTTON_JUMP] > 0.5f;

    // Add pressed buttons to the buffer
    buttons[game->frame] = (uint8_t)((inputs[BUTTON_RIGHT] << 0) & (inputs[BUTTON_LEFT] << 1) & (inputs[BUTTON_JUMP] << 2));

    /* printf("----------------------------\n");
    printf("Jump:\t%d\t%lf\n", inputs[BUTTON_JUMP], network_outputs[BUTTON_JUMP]);
    printf("Left:\t%d\t%lf\n", inputs[BUTTON_LEFT], network_outputs[BUTTON_LEFT]);
    printf("Right:\t%d\t%lf\n", inputs[BUTTON_RIGHT], network_outputs[BUTTON_RIGHT]);
    printf("----------------------------\n"); */

    if (game_update(game, player, inputs) == PLAYER_DEAD) {
        return -1;
    }

    return 0;
}

void write_out(uint8_t *buttons, size_t buttons_bytes, uint8_t *chrom, size_t chrom_bytes, unsigned int seed) {
    FILE *file = fopen("output.bin", "w");
    size_t total_bytes = buttons_bytes + chrom_bytes + sizeof(seed);

    //Seed
    fwrite(&seed, sizeof(unsigned int), 1, file);

    //Button presses
    fwrite(buttons, sizeof(uint8_t), buttons_bytes, file);

    //Chromosome
    fwrite(chrom, sizeof(uint8_t), chrom_bytes, file);

    printf("wrote %lu bytes\n", total_bytes);
    fclose(file);
}

uint extract_from_bytes(uint8_t *bytes, size_t nbytes, uint8_t *chrom, uint8_t *buttons) {
    uint seed = ((unsigned int *)bytes)[0];

    buttons = bytes + sizeof(unsigned int);
    chrom = chrom + MAX_FRAMES;
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
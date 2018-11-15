#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <defs.hpp>

#define IN_H 16
#define IN_W 16
#define BUTTON_COUNT 3
#define HLC 2
#define NPL 256

float vec_dot(float *a, float *b, int size);
float sigmoid(float x);
float soft_sign(float x);
int evaluate_frame(struct Game *game, struct Player *player, uint8_t *chrom, uint8_t *tiles, float *node_outputs, uint8_t *buttons);

#endif
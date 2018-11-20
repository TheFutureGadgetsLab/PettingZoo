#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

float vec_dot(float *a, float *b, int size);
float sigmoid(float x);
float soft_sign(float x);
int evaluate_frame(struct Game *game, struct Player *player, uint8_t *chrom, float *tiles, float *node_outputs, uint8_t *buttons);

#endif
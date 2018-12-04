#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <cuda_runtime.h>

__host__ __device__
float sigmoid(float x);
__host__ __device__
float soft_sign(float x);
__host__ __device__
int evaluate_frame(struct Game *game, struct Player *player, struct Chromosome *chrom, uint8_t *buttons, float *tiles, float *node_outputs);

#endif

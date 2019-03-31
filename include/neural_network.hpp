/**
 * @file neural_network.cpp
 * @author Benjamin Mastripolito, Haydn Jones
 * @brief  Defs for running a neural network 
 * @date 2018-12-11
 */
#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

float sigmoid(float x);
float soft_sign(float x);
int evaluate_frame(struct Game *game, struct Player *player, struct Chromosome *chrom, uint8_t *buttons, float *tiles, float *node_outputs);

#endif

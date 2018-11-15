#ifndef GENETIC_H
#define GENETIC_H

#include <stdint.h>

void single_point_breed(uint8_t *parentA, uint8_t *parentB, uint8_t *childA, uint8_t *childB, unsigned int seed);
void select_and_breed(uint8_t **generation, uint8_t **new_generation, unsigned int seed);

#endif
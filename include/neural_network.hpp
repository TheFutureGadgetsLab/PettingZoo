#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <defs.hpp>

#define IN_H 3
#define IN_W 3
#define BUTTON_COUNT 3
#define HLC 2
#define NPL 3

float vec_dot(float *a, float *b, int size);
float sigmoid(float x);
float soft_sign(float x);
uint8_t *extract_from_file(char *fname, uint8_t *buttons, uint *seed);


#endif
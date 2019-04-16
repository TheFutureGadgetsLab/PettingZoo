#ifndef RANDFUNCTS_H
#define RANDFUNCTS_H

int randint(unsigned int *seedp, int max);
int randrange(unsigned int *seedp, int min, int max);
int choose(unsigned int *seedp, int nargs, ...);
bool chance(unsigned int *seedState, float percent);

#endif
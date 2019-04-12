#include <randfuncts.hpp>
#include <stdlib.h>
#include <cstdarg>

/**
 * @brief Return an integer where 0 <= x <= max
 * 
 * @param seedp Reentrant seed pointer
 * @param max Max value for randint
 * @return int Random integer
 */
int randint(unsigned int *seedp, int max)
{
	return rand_r(seedp) % (max + 1);
}

/**
 * @brief Return an integer where min <= x <= max
 * 
 * @param seedp Reentrant seed pointer
 * @param min Min value in range
 * @param max Max value in range
 * @return int Random integer
 */
int randrange(unsigned int *seedp, int min, int max)
{
	if (min == max)
		return max;
	return min + (rand_r(seedp) % (abs(max - min) + 1));
}

/**
 * @brief Returns a random integer in list of integers
 * 
 * @param seedp Reentrant seed pointer
 * @param nargs Number of items to choose from
 * @param ... Items
 * @return int Chosen item
 */
int choose(unsigned int *seedp, int nargs, ...)
{
	va_list args;
	va_start(args, nargs);
	int array[nargs];
	int i;

	for (i = 0; i < nargs; i++) {
		array[i] = va_arg(args, int);
	}
	
	return array[randint(seedp, nargs - 1)];
}


/**
 * @brief Return 1 if random number is <= percent, otherwise 0
 * 
 * @param percent Percent chance between 0.0 and 100.0
 * @return int 1 or 0
 */
bool chance(float percent, unsigned int *seedState)
{
	return ((float)rand_r(seedState) / (float)RAND_MAX) < (percent / 100.0f);
}
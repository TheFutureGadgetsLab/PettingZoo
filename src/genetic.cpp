#include <genetic.hpp>
#include <stdlib.h>
#include <chromosome.hpp>
#include <string.h>
#include <stdio.h>

#define IN_W 3
#define IN_H 3
#define HLC  3
#define NPL  5

int rand_range(unsigned int *seedp, int min, int max);
void split(void *parentA, void *parentB, void *childA, void *childB, size_t length, size_t split);
int chance(unsigned int *seedp, double percent);

int main()
{
    uint8_t *parentA = NULL;
    uint8_t *parentB = NULL;
    uint8_t *childA = NULL;
    uint8_t *childB = NULL;

    parentA = (uint8_t *)malloc(sizeof(uint8_t) * get_chromosome_size_params(IN_H, IN_W, HLC, NPL));
    parentB = (uint8_t *)malloc(sizeof(uint8_t) * get_chromosome_size_params(IN_H, IN_W, HLC, NPL));
    childA = (uint8_t *)malloc(sizeof(uint8_t) * get_chromosome_size_params(IN_H, IN_W, HLC, NPL));
    childB = (uint8_t *)malloc(sizeof(uint8_t) * get_chromosome_size_params(IN_H, IN_W, HLC, NPL));

    generate_chromosome(parentA, IN_H, IN_W, HLC, NPL, 10);
    generate_chromosome(parentB, IN_H, IN_W, HLC, NPL, 27);

    puts("#########################################");
    puts("Parent A");
    print_chromosome(parentA);
    puts("Parent B");
    print_chromosome(parentB);
    puts("#########################################");

    single_point_breed(parentA, parentB, childA, childB, 12);

    puts("+++++++++++++++++++++++++++++++++++++++++");
    puts("Child A");
    print_chromosome(childA);
    puts("Child B");
    print_chromosome(childB);
    puts("+++++++++++++++++++++++++++++++++++++++++");
    
    return 0;   
}

// Select the best from a generation, then breed them into a new generation
void select_and_breed(uint8_t **generation, float *fitnesses, uint8_t **new_generation, size_t gen_size, uint seed)
{
    int i;
    float worst, best, average;
    uint8_t *survivors[gen_size / 2];
    size_t n_survivors = 0;
    uint seed_state = seed;

    //Find the worst and best score
    worst = best = fitnesses[0];
    for (i = 0; i < gen_size; i++) {
        if (fitnesses[i] < worst) {
            worst = fitnesses[i];
        }
        if (fitnesses[i] > best) {
            best = fitnesses[i];
        }
    }
    average = (best - worst) / 2;

    //Select survivors
    while (n_survivors < gen_size / 2) {
        i = rand_r(&seed_state) % gen_size;
        if (chance(&seed_state, fitnesses[i] / best)) {
            survivors[n_survivors] = generation[i];
            n_survivors += 1;
        }
    }

    //Breed
    //TODO
}

// Chooses as single point in each section of the chromosomes and childA is set up as
// ParentA | Parent B, while childB is set up as ParentB | ParentA
void single_point_breed(uint8_t *parentA, uint8_t *parentB, uint8_t *childA, uint8_t *childB, unsigned int seed)
{   
    unsigned int seed_state = seed;
    size_t section_size;
    int split_loc, hl;
    uint8_t *cur_section_u, *next_section_u;
    float *cur_section_f, *next_section_f;

    // Set chrom headers
    childA[0] = parentA[0];
    childB[0] = parentA[0];
    childA[1] = parentA[1];
    childB[1] = parentA[1];
    *((uint16_t *)childA + 1) = *((uint16_t *)parentA + 1);
    *((uint16_t *)childB + 1) = *((uint16_t *)parentA + 1);
    childA[4] = parentA[4];
    childB[4] = parentA[4];

    // Cross input layers
    cur_section_u = locate_input_act(parentA);
    next_section_f = locate_input_adj(parentA);
    section_size = (uint8_t*)next_section_f - cur_section_u;
    split_loc = rand_r(&seed_state) % (section_size + 1);
    split(locate_input_act(parentA), locate_input_act(parentB), 
          locate_input_act(childA),  locate_input_act(childB), section_size, split_loc);

    // Cross input adj layers
    cur_section_f = locate_input_adj(parentA);
    next_section_u = locate_hidden_act(parentA);
    section_size = next_section_u - (uint8_t*)cur_section_f;
    split_loc = rand_r(&seed_state) % ((section_size + 1) / 4);
    split(locate_input_adj(parentA), locate_input_adj(parentB), 
            locate_input_adj(childA), locate_input_adj(childB), section_size, split_loc * 4);

    // Cross hidden act layer
    cur_section_u = locate_hidden_act(parentA);
    next_section_f = locate_hidden_adj(parentA, 0);
    section_size = (uint8_t*)next_section_f - cur_section_u;
    split_loc = rand_r(&seed_state) % (section_size + 1);
    split(locate_hidden_act(parentA), locate_hidden_act(parentB), 
          locate_hidden_act(childA),  locate_hidden_act(childB), section_size, split_loc);
    
    // Cross hidden layers
    for (hl = 0; hl < parentA[4] - 1; hl++) {
        cur_section_f = locate_hidden_adj(parentA, hl);
        next_section_f = locate_hidden_adj(parentA, hl + 1);
        if (next_section_f == NULL)
            next_section_f = locate_out_adj(parentA);
        section_size = (uint8_t*)next_section_f - (uint8_t*)cur_section_f;
        split_loc = rand_r(&seed_state) % ((section_size + 1) / 4);
        split(locate_hidden_adj(parentA, hl), locate_hidden_adj(parentB, hl), 
                locate_hidden_adj(childA, hl), locate_hidden_adj(childB, hl), section_size, split_loc * 4);
    }

    // Cross final section
    cur_section_f = locate_out_adj(parentA);
    next_section_u = parentA + get_chromosome_size(parentA);
    section_size = next_section_u - (uint8_t*)cur_section_f;
    split_loc = rand_r(&seed_state) % ((section_size + 1) / 4);
    printf("Split loc: %d\n", split_loc);
    split(locate_out_adj(parentA), locate_out_adj(parentB), 
            locate_out_adj(childA), locate_out_adj(childB), section_size, split_loc * 4);
}

void split(void *parentA, void *parentB, void *childA, void *childB, size_t length, size_t split)
{
    memcpy(childA, parentA, split);
    memcpy((uint8_t *)childA + split, (uint8_t *)parentB + split, length - split);

    memcpy(childB, parentB, split);
    memcpy((uint8_t *)childB + split, (uint8_t *)parentA + split, length - split);
}

// Return 0 or 1 probabilistically
int chance(unsigned int *seedp, double percent) {
	return ((double)rand_r(seedp) / (double)RAND_MAX) <= (percent / 100.0);
}

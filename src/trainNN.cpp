#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <sys/stat.h>
#include <genetic.hpp>
#include <chromosome.hpp>
#include <neural_network.hpp>
#include <gamelogic.hpp>
#include <levelgen.hpp>

void print_gen_stats(struct Player players[GEN_SIZE], int quiet);
void write_out_progress(FILE *fh, struct Player players[GEN_SIZE]);
FILE* create_output_dir(unsigned int seed);
void write_out(char *name, uint8_t *buttons, size_t buttons_bytes, uint8_t *chrom, unsigned int seed);

int main()
{
    struct Game *games;
    struct Player *players;
    uint8_t *genA[GEN_SIZE], *genB[GEN_SIZE];
    uint8_t **cur_gen, **next_gen, **tmp;
    int gen, game;
    unsigned int seed, level_seed;
    struct RecordedChromosome winner;
    FILE *out_file;
    char fname[4096];

    winner.fitness = 0;

    seed = (unsigned int)time(NULL);
    srand(seed);

    out_file = create_output_dir(seed);

    // Arrays for game and player structures
    games = (struct Game *)malloc(sizeof(struct Game) * GEN_SIZE);
    players = (struct Player *)malloc(sizeof(struct Player) * GEN_SIZE);

    printf("Running with %d chromosomes for %d generations\n", GEN_SIZE, GENERATIONS);
    printf("Chromosome stats:\n");
    printf("  IN_H: %d\n  IN_W: %d\n  HLC: %d\n  NPL: %d\n", IN_H, IN_W, HLC, NPL);
    printf("Seed: %u\n", seed);

    // Allocate space for genA and genB chromosomes
    for (game = 0; game < GEN_SIZE; game++) {
        genA[game] = (uint8_t *)malloc(sizeof(uint8_t) * get_chromosome_size_params(IN_H, IN_W, HLC, NPL));
        genB[game] = (uint8_t *)malloc(sizeof(uint8_t) * get_chromosome_size_params(IN_H, IN_W, HLC, NPL));
        // Generate random chromosomes
        generate_chromosome(genA[game],  IN_H, IN_W, HLC, NPL, rand());
    }

    // Level seed for entire run
    level_seed = rand();

    // Initially point current gen to genA, then swap next gen
    cur_gen = genA;
    next_gen = genB;
    for (gen = 0; gen < GENERATIONS; gen++) {
        puts("----------------------------");
        printf("Running generation %d/%d\n", gen, GENERATIONS);

        // Generate seed for this gens levels and generate them
        for (game = 0; game < GEN_SIZE; game++) {
            game_setup(&players[game]);
            levelgen_gen_map(&games[game], level_seed);
        }

        run_generation(games, players, cur_gen, &winner);

        // Get stats from run (1 tells function to not print each players fitness)
        print_gen_stats(players, 1);

        // Write out progress
        write_out_progress(out_file, players);

        // Usher in the new generation
        select_and_breed(players, cur_gen, next_gen);

        // Point current gen to new chromosomes and next gen to old
        tmp = cur_gen;
        cur_gen = next_gen;
        next_gen = tmp;
    }
    puts("----------------------------\n");

    printf("Best chromosome:\n");
    printf("  Fitness: %f\n  Seed: %u\n", winner.fitness, winner.game->seed);

    // Write out best chromosome
    sprintf(fname, "./%d/best.bin", seed);
    write_out(fname, winner.buttons, MAX_FRAMES, winner.chrom, winner.game->seed);

    for (game = 0; game < GEN_SIZE; game++) {
        free(genA[game]);
        free(genB[game]);
    }

    free(games);
    free(players);
    fclose(out_file);

    return 0;
}

void print_gen_stats(struct Player players[GEN_SIZE], int quiet)
{
    int game, completed, timedout, died;
    float max, min, avg;

    completed = 0;
    timedout = 0;
    died = 0;
    avg = 0.0f;
    max = players[0].fitness;
    min = players[0].fitness;
    for(game = 0; game < GEN_SIZE; game++) {
        avg += players[game].fitness;

        if (players[game].fitness > max)
            max = players[game].fitness;
        else if (players[game].fitness < min)
            min = players[game].fitness;

        if (players[game].death_type == PLAYER_COMPLETE)
            completed++;
        else if (players[game].death_type == PLAYER_TIMEOUT)
            timedout++;
        else if (players[game].death_type == PLAYER_DEAD)
            died++;

        if (!quiet)
            printf("Player %d fitness: %0.2lf\n", game, players[game].fitness);
    }

    avg /= GEN_SIZE;

    printf("\nDied:        %.2lf%%\nTimed out:   %.2lf%%\nCompleted:   %.2lf%%\nAvg fitness: %.2lf\n",
            (float)died / GEN_SIZE * 100, (float)timedout / GEN_SIZE * 100,
            (float)completed / GEN_SIZE * 100, avg);
    printf("Max fitness: %.2lf\n", max);
    printf("Min fitness: %.2lf\n", min);
}

void write_out_progress(FILE *fh, struct Player players[GEN_SIZE])
{
    int game, completed, timedout, died;
    float max, min, avg;

    completed = 0;
    timedout = 0;
    died = 0;
    avg = 0.0f;
    max = players[0].fitness;
    min = players[0].fitness;
    for(game = 0; game < GEN_SIZE; game++) {
        avg += players[game].fitness;

        if (players[game].fitness > max)
            max = players[game].fitness;
        else if (players[game].fitness < min)
            min = players[game].fitness;

        if (players[game].death_type == PLAYER_COMPLETE)
            completed++;
        else if (players[game].death_type == PLAYER_TIMEOUT)
            timedout++;
        else if (players[game].death_type == PLAYER_DEAD)
            died++;
    }
    avg /= GEN_SIZE;

    fprintf(fh, "%d, %d, %d, %lf, %lf, %lf\n", completed, timedout, died, avg, max, min);
    fflush(fh);
}

FILE* create_output_dir(unsigned int seed)
{
    FILE* out_file;
    char name[4069];

    sprintf(name, "./%d", seed);

    mkdir(name, S_IRWXU | S_IRWXG | S_IRWXO);

    sprintf(name, "./%d/run_data.txt", seed);
    out_file = fopen(name, "w+");

    // Write out header data
    fprintf(out_file, "%d, %d, %d, %d, %d, %d, %lf\n", IN_H, IN_W, HLC, NPL, GEN_SIZE, GENERATIONS, MUTATE_RATE);
    fflush(out_file);

    return out_file;
}

void write_out(char *name, uint8_t *buttons, size_t buttons_bytes, uint8_t *chrom, unsigned int seed)
{
    FILE *file = fopen(name, "wb");

    size_t chrom_bytes = get_chromosome_size(chrom);

    //Seed
    fwrite(&seed, sizeof(unsigned int), 1, file);

    //Button presses
    fwrite(buttons, sizeof(uint8_t), buttons_bytes, file);

    //Chromosome
    fwrite(chrom, sizeof(uint8_t), chrom_bytes, file);

    fclose(file);
}
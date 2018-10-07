/*
Five stages of genetic algo;
    1. Chromosome Encoding
    2. Fitness function
    3. Selection
    4. Recombination
    5. Evolution

Chromosome encoding:
    GA manipulates a population of chromosomes, which are representations of a solution to the problem. A particular location in
    the chromosome string is referred to as a gene and the value at said location is referred to as an allele. Chromosomes of 
    length 100 or more are not uncommon, resulting in 2^100 - 10^30 chromosomes.
Fitness function:
    Evaluates how well a chromosome solves the problem. Chromosome can be described as the genotype and the solution from the
    chromosome is the phenotype.
Selction:
    Fitness is used as a descriminator for selection. Individuals with higher fitness are more likely to reproduce, which creates
    a selective pressure towards higher-fit populations. Highly-fit chromosomes also have a possibility of reproducing multiple
    times or even with themselves. Various selection algorithms:
        - Roulette Wheel: 
*/
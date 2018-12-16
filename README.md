# Petting Zoo
CUDA and serial implementation of a genetic algorithm learning to player a game.

## Game
The game is a simple platformer.

## Algorithms
Evolution, Neural Networks, Stuff.

## Installation and Requirements
SFML and CUDA must be installed to build. CUDA is only required to run on the GPU and SFML is only required to watch a chromosome play.
Here are some sample build instructions:

```
mkdir build
cd build
cmake ..
make 
```
## Running
When built the following executables will be created:
- pettingzoo: This is the platformer, just run it to play the game. To watch a chromosome play, run `./pettingzoo -f PATH_TO_CHROMOSOME`
- trainCPU: This runs the genetic algorithm on the CPU. Run `./trainCPU -h` for help
- trainGPU: This runs the genetic algorithm on the GPU. Run `./trainGPU -h` for help

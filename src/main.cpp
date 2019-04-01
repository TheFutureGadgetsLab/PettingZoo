#include <SFML/Graphics.hpp>
#include <defs.hpp>
#include <rendering.hpp>
#include <gamelogic.hpp>
#include <sys/stat.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <levelgen.hpp>
#include <chromosome.hpp>
#include <neural_network.hpp>

#define RIGHT(x) ((x >> 0) & 0x1)
#define LEFT(x)  ((x >> 1) & 0x1)
#define JUMP(x)  ((x >> 2) & 0x1)

#define GAME_EXIT  1
#define GAME_RESET 2
#define GAME_NEW   3

void reset_game(struct Game& game, struct Player& player, unsigned int seed);
int get_player_input(sf::RenderWindow &window, uint8_t inputs[BUTTON_COUNT], bool *draw_overlay);

int main(int argc, char **argv)
{
    struct Params params = {IN_H, IN_W, HLC, NPL, GEN_SIZE, GENERATIONS, MUTATE_RATE};
	bool draw_overlay, replay_ai;
	int opt, ret;
	uint8_t inputs[BUTTON_COUNT] = {0};
	float *input_tiles = NULL;
	float *node_outputs = NULL;
	struct Chromosome chrom;
	unsigned int seed;
	struct Game game;
	struct Player player;
	sf::RenderWindow window(sf::VideoMode(800, 600), "PettingZoo");
	sf::Color bg_color(135, 206, 235);

	seed = time(NULL);
	replay_ai = false;
	while ((opt = getopt(argc, argv, "hf:")) != -1) {
		switch (opt) {
		// Read in replay file to watch NN
		case 'f':
			replay_ai = true;
			seed = extract_from_file(optarg, &chrom);
			
			input_tiles = (float *)malloc(sizeof(float) * chrom.in_w * chrom.in_h);
			node_outputs = (float *)malloc(sizeof(float) * chrom.npl * chrom.hlc);
			params.in_w = chrom.in_w;
			params.in_h = chrom.in_h;
			params.hlc = chrom.hlc;
			params.npl = chrom.npl;
			break;
		default:
			printf("Usage: %s [-f PATH_TO_CHROMOSOME]\n", argv[0]);
			printf("Buttons:\n");
			printf("  r:    Resets game state and takes you to the beginning of the level\n");
			printf("  n:    Generates new level, resets game state, and takes you to the beginning\n");
			printf("  o:    Opens debug overlay and displays input grid of chromosome. Does not currently reflect chromosomes actual input size\n");
			return 0;
		}
	}

	window.setKeyRepeatEnabled(false);
	window.setVerticalSyncEnabled(true);
		
	game_setup(game, seed);
	player_setup(player);
	render_load_assets();
	render_gen_map(game);

	draw_overlay = false;
	while (window.isOpen()) {
		// Get player input
		ret = get_player_input(window, inputs, &draw_overlay);
		if (ret == GAME_RESET) {
			reset_game(game, player, seed);
		} else if (ret == GAME_NEW) {
			seed = rand();
			reset_game(game, player, seed);
		} else if (ret == GAME_EXIT) {
			break;
		}

		//Get buttons if replaying NN
		if (replay_ai) {
			evaluate_frame(game, player, chrom, inputs, input_tiles, node_outputs);
		}

		ret = game_update(game, player, inputs);

		if (ret == PLAYER_DEAD || ret == PLAYER_TIMEOUT) {
			if (ret == PLAYER_DEAD)
				printf("Player died\n");
			else
				printf("Player timed out\n");
    		printf("Fitness: %0.2lf\n", player.fitness);

			reset_game(game, player, seed);
		} else if (ret == REDRAW) {
			render_gen_map(game);
		}

		// Update camera
		render_handle_camera(window, player);

		//Clear the screen
		window.clear(bg_color);

		//Draw background, tiles, and entities
		render_draw_state(window, game, player);

		//Draw debug overlay
		if (draw_overlay) {
			render_debug_overlay(window, game, player);
		}

		// Score and time
		render_hud(window, player, inputs);

		window.display();
	}

	if (replay_ai) {
		free(input_tiles);
		free(node_outputs);
		free_chromosome(&chrom);
	}
	
	window.close();

	return 0;
}

int get_player_input(sf::RenderWindow &window, uint8_t inputs[BUTTON_COUNT], bool *draw_overlay)
{
	sf::Event event;
	while (window.pollEvent(event)) {
		int is_pressed = (event.type == sf::Event::KeyPressed);
		switch(event.key.code) {
			// Jump
			case sf::Keyboard::Up:
			case sf::Keyboard::Space:
			case sf::Keyboard::W:
				inputs[BUTTON_JUMP] = is_pressed;
				break;
			// Left
			case sf::Keyboard::Left:
			case sf::Keyboard::A:
				inputs[BUTTON_LEFT] = is_pressed;
				break;
			// Right
			case sf::Keyboard::Right:
			case sf::Keyboard::D:
				inputs[BUTTON_RIGHT] = is_pressed;
				break;
			case sf::Keyboard::Escape:
				return GAME_EXIT;
			case sf::Keyboard::O:
				*draw_overlay = *draw_overlay ^ is_pressed;
				break;
			// Reset game state
			case sf::Keyboard::R:
				if (is_pressed)
					return GAME_RESET;
				break;
			// New game state
			case sf::Keyboard::N:
				if (is_pressed)
					return GAME_NEW;
				break;
			default:
				break;
		}

		if (event.type == sf::Event::Closed) {
			window.close();
		} else if (event.type == sf::Event::Resized) {
			render_scale_window(window, event);
		}
	}

	return 0;
}

void reset_game(struct Game& game, struct Player& player, unsigned int seed)
{
	game_setup(game, seed);
	player_setup(player);
	render_gen_map(game);
}

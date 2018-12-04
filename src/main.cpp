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

void reset_game(struct Game *game, struct Player *player, unsigned int seed, bool replay_ai);
int get_player_input(sf::RenderWindow &window, uint8_t inputs[BUTTON_COUNT], bool *draw_overlay);

int main(int argc, char **argv)
{
	bool draw_overlay, replay_ai;
	int opt, ret;
	uint8_t inputs[BUTTON_COUNT] = {0};
	uint8_t buttons;
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
	while ((opt = getopt(argc, argv, "f:")) != -1) {
		switch (opt) {
		// Read in replay file to watch NN
		case 'f':
			replay_ai = true;
			
			input_tiles = (float *)malloc(sizeof(float) * IN_W * IN_H);
			node_outputs = (float *)malloc(sizeof(float) * NPL * HLC);

			seed = extract_from_file(optarg, &chrom);
			printf("Seed: %u\n", seed);
			break;
		default:
			printf("Usage: %s [-f replayfile]\n", argv[0]);
			exit(EXIT_FAILURE);
		}
	}

	window.setKeyRepeatEnabled(false);
	window.setVerticalSyncEnabled(true);
		
	game_setup(&game, seed);
	player_setup(&player);
	render_load_assets();
	render_gen_map(game);

	draw_overlay = false;
	while (window.isOpen()) {
		// Get player input
		ret = get_player_input(window, inputs, &draw_overlay);
		if (ret == GAME_RESET) {
			reset_game(&game, &player, seed, replay_ai);
		} else if (ret == GAME_EXIT) {
			window.close();
			return 0;
		}

		//Get buttons and update game state
		if (replay_ai) {
			ret = evaluate_frame(&game, &player, &chrom, &buttons, input_tiles, node_outputs);

			inputs[BUTTON_RIGHT] = RIGHT(buttons);
			inputs[BUTTON_LEFT] =  LEFT(buttons);
			inputs[BUTTON_JUMP] = JUMP(buttons);
		} else {
			ret = game_update(&game, &player, inputs);
		}

		if (ret == PLAYER_DEAD || ret == PLAYER_TIMEOUT) {
			if (ret == PLAYER_DEAD)
				printf("Player died\n");
			else
				printf("Player timed out\n");
    		printf("Fitness: %0.2lf\n", player.fitness);

			reset_game(&game, &player, seed, replay_ai);
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
				if (is_pressed) {
					return GAME_RESET;
				}
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

void reset_game(struct Game *game, struct Player *player, unsigned int seed, bool replay_ai)
{
	if (!replay_ai)
		seed = time(NULL);
	
	game_setup(game, seed);
	player_setup(player);
	render_gen_map(*game);
}

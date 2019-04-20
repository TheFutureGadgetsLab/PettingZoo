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
#include <NeuralNetwork.hpp>

#define GAME_EXIT  1
#define GAME_RESET 2
#define GAME_NEW   3

void reset_game(Game& game, Player& player, unsigned int seed);
int get_player_input(sf::RenderWindow &window, Player& player, bool *draw_overlay);

int main(int argc, char **argv)
{
	unsigned int seed;
    Params params;
	bool draw_overlay, replay_ai;
	int opt, ret;
	NeuralNetwork *chrom;
	Game game;
	Player player;
	sf::RenderWindow window(sf::VideoMode(800, 600), "PettingZoo");
	sf::Color bg_color(135, 206, 235);

	seed = time(NULL);

	replay_ai = false;
	while ((opt = getopt(argc, argv, "hf:")) != -1) {
		switch (opt) {
		// Read in replay file to watch NN
		case 'f':
			replay_ai = true;
			// seed = getStatsFromFile(optarg, params);
			// chrom = new Chromosome(optarg);
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
	
	render_load_assets();
	reset_game(game, player, seed);

	draw_overlay = false;

	while (window.isOpen()) {
		// Get player input
		ret = get_player_input(window, player, &draw_overlay);
		if (ret == GAME_RESET) {
			reset_game(game, player, seed);
		} else if (ret == GAME_NEW) {
			seed = rand();
			reset_game(game, player, seed);
		} else if (ret == GAME_EXIT) {
			break;
		}
		
		//Get buttons if replaying NN
		// if (replay_ai) {
 			// evaluate_frame(game, player, *chrom);
		// }

		ret = game.update(player);

		if (ret == PLAYER_DEAD || ret == PLAYER_TIMEOUT || ret == PLAYER_COMPLETE) {
			if (ret == PLAYER_DEAD)
				printf("Player died\n");
			else if (ret == PLAYER_TIMEOUT)
				printf("Player timed out\n");
			else if (ret == PLAYER_COMPLETE)
				printf("Player won!\n");

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
		render_hud(window, player);

		window.display();
	}

	if (replay_ai) {
		delete chrom;
	}
	
	window.close();

	return 0;
}

int get_player_input(sf::RenderWindow &window, Player& player, bool *draw_overlay)
{
	sf::Event event;
	while (window.pollEvent(event)) {
		int is_pressed = (event.type == sf::Event::KeyPressed);
		switch(event.key.code) {
			// Jump
			case sf::Keyboard::Up:
			case sf::Keyboard::Space:
			case sf::Keyboard::W:
				player.jump = is_pressed;
				break;
			// Left
			case sf::Keyboard::Left:
			case sf::Keyboard::A:
				player.left = is_pressed;
				break;
			// Right
			case sf::Keyboard::Right:
			case sf::Keyboard::D:
				player.right = is_pressed;
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

void reset_game(Game& game, Player& player, unsigned int seed)
{
	game.genMap(seed);
	player.reset();
	render_gen_map(game);
}

#include <SFML/Graphics.hpp>
#include <defs.hpp>
#include <rendering.hpp>
#include <gamelogic.hpp>
#include <sys/stat.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <levelgen.hpp>

#define RIGHT(x) ((x >> 0) & 0x1)
#define LEFT(x)  ((x >> 1) & 0x1)
#define JUMP(x)  ((x >> 2) & 0x1)

#define GAME_EXIT  1
#define GAME_RESET 2

uint8_t *extract_from_file(const char *fname, uint8_t *buttons, unsigned int *seed);
void reset_game(struct Game *game, struct Player *player, unsigned int seed, bool replay_ai);
int get_player_input(sf::RenderWindow &window, uint8_t inputs[BUTTON_COUNT], bool *draw_overlay);

int main(int argc, char **argv)
{
	bool draw_overlay, replay_ai;
	int opt, ret, frame;
	uint8_t inputs[BUTTON_COUNT] = {0};
	uint8_t buttons[MAX_FRAMES];
	uint8_t *chrom;
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
			chrom = extract_from_file(optarg, buttons, &seed);
			free(chrom);
			break;
		default:
			printf("Usage: %s [-f replayfile]\n", argv[0]);
			exit(EXIT_FAILURE);
		}
	}

	window.setKeyRepeatEnabled(false);
	window.setVerticalSyncEnabled(true);
		
	game_setup(&player);
	levelgen_gen_map(&game, seed);
	render_load_assets();
	render_gen_map(game);

	frame = 0;
	draw_overlay = false;
	while (window.isOpen()) {
		// Get player input
		ret = get_player_input(window, inputs, &draw_overlay);
		if (ret == GAME_RESET) {
			frame = 0;
			reset_game(&game, &player, seed, replay_ai);
		} else if (ret == GAME_EXIT) {
			window.close();
			return 0;
		}

		//Get buttons
		if (replay_ai) {
			inputs[BUTTON_RIGHT] = RIGHT(buttons[frame]);
			inputs[BUTTON_LEFT] =  LEFT(buttons[frame]);
			inputs[BUTTON_JUMP] = JUMP(buttons[frame]);
			frame++;
		}

		//Update game state
		ret = game_update(&game, &player, inputs);
		if (ret == PLAYER_DEAD || ret == PLAYER_TIMEOUT) {
			if (ret == PLAYER_DEAD)
				printf("Player died\n");
			else
				printf("Player timed out\n");
    		printf("Fitness: %0.2lf\n", player.fitness);

			frame = 0;
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
	
	game_setup(player);
	levelgen_gen_map(game, seed);
	render_gen_map(*game);
}

/*
 * Extracts seed, chromosome, and button presses from file.
 * THIS FUNCTION RETURNS A POINTER TO THE EXTRACTED
 * CHROMOSOME THAT ***YOU MUST FREE YOURSELF***
 */
uint8_t *extract_from_file(const char *fname, uint8_t *buttons, unsigned int *seed)
{
    FILE *file = NULL;
    uint8_t *data = NULL;
    uint8_t *chrom;
    size_t file_size, chrom_size, read;
    struct stat st;

    if (stat(fname, &st) == -1) {
		printf("Error reading file '%s'!\n", fname);
		exit(EXIT_FAILURE);
	}

	file_size = st.st_size;

    file = fopen(fname, "rb");

	data = (uint8_t *)malloc(sizeof(uint8_t) * (file_size + 1));
    read = fread(data, sizeof(uint8_t), file_size, file);

	if (read != file_size) {
		printf("Error reading file!\n");
		exit(EXIT_FAILURE);
	}

    chrom_size = file_size - MAX_FRAMES - sizeof(unsigned int);
    chrom = (uint8_t *)malloc(chrom_size);

    // Populate button array
    memcpy(buttons, data + sizeof(unsigned int), MAX_FRAMES);
    // Populate chromosome
    memcpy(chrom, data + sizeof(unsigned int) + MAX_FRAMES, chrom_size);

    *seed = ((unsigned int *)data)[0];

    free(data);

    return chrom;
}

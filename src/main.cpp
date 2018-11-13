#include <SFML/Graphics.hpp>
#include <defs.hpp>
#include <rendering.hpp>
#include <gamelogic.hpp>
#include <sys/stat.h>
#include <time.h>
#include <string.h>

uint8_t *extract_from_file(char *fname, uint8_t *buttons, uint *seed);

int main()
{
	int draw_overlay = 0;
	int input[BUTTON_COUNT] = {0};
	int ret, frame;
    sf::RenderWindow window(sf::VideoMode(800, 600), "PettingZoo");
	sf::Time frame_time;
	sf::Clock clock;
	sf::Color bg_color(135, 206, 235);
	struct Game game;
	struct Player player;
	uint8_t buttons[MAX_FRAMES];
	unsigned int seed;

	seed = time(NULL);

	window.setKeyRepeatEnabled(false);
	window.setVerticalSyncEnabled(true);

	game_setup(&game, &player, seed);
	render_load_assets();
	render_gen_map(game);

	frame = 0;
	while (window.isOpen()) {
		sf::Event event;
		while (window.pollEvent(event)) {
			int is_pressed = (event.type == sf::Event::KeyPressed);
			switch(event.key.code) {
				// Jump
				case sf::Keyboard::Up:
				case sf::Keyboard::Space:
				case sf::Keyboard::W:
					input[BUTTON_JUMP] = is_pressed;
					break;
				// Left
				case sf::Keyboard::Left:
				case sf::Keyboard::A:
					input[BUTTON_LEFT] = is_pressed;
					break;
				// Right
				case sf::Keyboard::Right:
				case sf::Keyboard::D:
					input[BUTTON_RIGHT] = is_pressed;
					break;
				case sf::Keyboard::Escape:
					return 0;
				case sf::Keyboard::O:
					draw_overlay ^= 1 * is_pressed;
					break;
				// Reset game state
				case sf::Keyboard::R:
					if (is_pressed) {
						seed = time(NULL);
						game_setup(&game, &player, seed);
						render_gen_map(game);
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

		// //Get buttons
		// curbuttons = buttons[frame];
		// input[BUTTON_RIGHT] = curbuttons & 0x1;
		// input[BUTTON_LEFT] = curbuttons & 0x2;
		// input[BUTTON_JUMP] = curbuttons & 0x4;

		//Update game state
		ret = game_update(&game, &player, input);
		if (ret == PLAYER_DEAD || ret == PLAYER_TIMEOUT) {
			if (ret == PLAYER_DEAD)
				printf("Player died:\n");
			else
				printf("Player timed out:\n");
    		printf("Fitness: %d\n", player.fitness);

			seed = time(NULL);
			game_setup(&game, &player, seed);
			render_gen_map(game);
		} else if (ret == REDRAW) {
			render_gen_map(game);
		}

		// Update camera
		render_handle_camera(window, player);

		//Clear the screen
		window.clear(bg_color);

		//Draw background, tiles, and entities
		render_draw_state(window, game, player);

		//Draw debug overlay + fps
		frame_time = clock.getElapsedTime();
		clock.restart();
		if (draw_overlay) {
			render_debug_overlay(window, game, player, frame_time);
		}

		// Score and time
		render_hud(window, player, input);

		window.display();
		frame++;
	}

	return 0;
}

/*
 * Extracts seed, chromosome, and button presses from file.
 * THIS FUNCTION RETURNS A POINTER TO THE EXTRACTED
 * CHROMOSOME THAT ***YOU MUST FREE YOURSELF***
 */
uint8_t *extract_from_file(char *fname, uint8_t *buttons, uint *seed)
{
    FILE *file = NULL;
    uint8_t *data = NULL;
    uint8_t *chrom;
    size_t file_size, chrom_size;
    struct stat st;

    stat(fname, &st);
	file_size = st.st_size;

    file = fopen(fname, "r");

	data = (uint8_t *)malloc(file_size);
    fread(data, sizeof(uint8_t), file_size, file);

    chrom_size = file_size - MAX_FRAMES - sizeof(uint);
    chrom = (uint8_t *)malloc(chrom_size);

    // Populate button array
    memcpy(buttons, data + sizeof(uint), MAX_FRAMES);
    // Populate chromosome
    memcpy(chrom, data + sizeof(uint) + MAX_FRAMES, chrom_size);

    *seed = ((unsigned int *)data)[0];

    free(data);

    return chrom;
}
#include <SFML/Audio.h>
#include <SFML/Graphics.h>
#include <SFML/Config.h>
#include <SFML/System.h>
#include <stdio.h>
#include <stdlib.h>
#include <game.h>
#include <defs.h>
#include <unistd.h>

float zoom = 2.0;

int rescale_window(sfView *view, sfEvent event);

int main(int argc, char **argv)
{
	int opt, draw_overlay;

	sfVideoMode mode = {800, 600, 32};
	sfRenderWindow* window;
	sfVector2f moveby = {0, 0};
	sfEvent event;
	sfTime time;
	sfClock *clock;
	sfView *view;
	sfColor background = {230, 230, 230, 1};

	draw_overlay = 0;
	// while ((opt = getopt(argc, argv, "o")) != -1) {
	// 	switch (opt) {
	// 		case 'o':
	// 			draw_overlay = 1;
	// 			break;
	// 		default:
	// 			printf("Usage: ./pettingzoo [-g] [-c]\n");
	// 			printf("\t-o Display overlay\n");
	// 			exit(EXIT_FAILURE);
	// 	}
	// }

	// Create the clock
	clock = sfClock_create();

	// Create the main window
	window = sfRenderWindow_create(mode, "Petting Zoo", sfResize | sfClose, NULL);
	if (!window)
		return -1;
	sfRenderWindow_setKeyRepeatEnabled(window, sfFalse);

	// Vsync 
	sfRenderWindow_setVerticalSyncEnabled(window, sfTrue);

	//Load assets
	game_load_assets(draw_overlay);

	//Generate game
	game_gen_map();
	game_setup();

	// Start the game loop
	view = sfView_copy(sfRenderWindow_getView(window));
	while (sfRenderWindow_isOpen(window))
	{
		// Process events
		while (sfRenderWindow_pollEvent(window, &event))
		{
			// Close window
			if (event.type == sfEvtClosed) {
				sfRenderWindow_close(window);
			} else if (event.type == sfEvtResized) {
				rescale_window(view, event);
			} else if (event.type == sfEvtKeyPressed) {
				if (event.key.code == sfKeyUp) {
					player.jump = 1;
				} else if (event.key.code == sfKeyLeft) {
					player.left = 1;
				}  else if (event.key.code == sfKeyRight) {
					player.right = 1;
				} else if (event.key.code == sfKeyEscape) {
					goto exit;
				} else if (event.key.code == sfKeyO) {
					draw_overlay ^= 1;
				}
			} else if (event.type == sfEvtKeyReleased) {
				if (event.key.code == sfKeyLeft) {
					player.left = 0;
				}  else if (event.key.code == sfKeyRight) {
					player.right = 0;
				}
			}
		}
		// Frametime
		time = sfClock_getElapsedTime(clock);

		game_update();

		// Get coords
		sfVector2f center = sfView_getCenter(view);
		sfVector2f size = sfView_getSize(view);
		sfVector2f viewpos;
		viewpos.x = center.x - (size.x / 2.0);
		viewpos.y = center.y - (size.y / 2.0);

		// Move camera towards player position
		sfVector2f moveto = {player.position.x + 16, player.position.y + 16};
		sfView_setCenter(view, moveto);
		sfRenderWindow_setView(window, view);

		// Clear the screen
		sfRenderWindow_clear(window, background);

		// Draw background
		game_draw_other(window, view);

		//Draw the tiles and entities
		game_draw_tiles(window, view, draw_overlay);
		game_draw_entities(window, view);

		// Draw coords if needed
		if (draw_overlay) {
			game_draw_overlay_text(window, view, time);
		}

		// Restart the clock
		sfClock_restart(clock);
		// Update the window
		sfRenderWindow_display(window);
	}

// Goto for leaving event and game loop if escape is hit, break wont work
// because of nested loops.
exit:
	// Cleanup resources
	sfRenderWindow_destroy(window);
	sfClock_destroy(clock);
	sfView_destroy(view);
	
	return 0;
}

int rescale_window(sfView *view, sfEvent event)
{
	sfVector2f win_size = {event.size.width, event.size.height};
	sfVector2f win_center = sfView_getCenter(view);
	win_size.x /= zoom;
	win_size.y /= zoom;
	
	sfView_setSize(view, win_size);

	return 0;
}
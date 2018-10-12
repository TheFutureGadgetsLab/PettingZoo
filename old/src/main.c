#include <SFML/Graphics.h>
#include <stdio.h>
#include <stdlib.h>
#include <rendering.h>
#include <defs.h>
#include <gamelogic.h>

int main(int argc, char **argv)
{
	int draw_overlay = 0;
	int input[BUTTON_COUNT] = {0};

	sfVideoMode mode = {800, 600, 32};
	sfRenderWindow* window;
	sfEvent event;
	sfTime time;
	sfClock *clock;
	sfView *view;

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
	render_load_assets(draw_overlay);

	//Generate game
	game_setup();

	// Start the game loop
	view = sfView_copy(sfRenderWindow_getView(window));
	while (sfRenderWindow_isOpen(window))
	{
		// Process events
		while (sfRenderWindow_pollEvent(window, &event))
		{
			if (event.type == sfEvtKeyPressed) {
				if (event.key.code == sfKeyUp || event.key.code == sfKeySpace) {
					input[BUTTON_JUMP] = 1;
				} else if (event.key.code == sfKeyLeft) {
					input[BUTTON_LEFT] = 1;
				} else if (event.key.code == sfKeyRight) {
					input[BUTTON_RIGHT] = 1;
				} else if (event.key.code == sfKeyEscape) {
					goto exit;
				} else if (event.key.code == sfKeyO) {
					draw_overlay ^= 1;
				}
			} else if (event.type == sfEvtKeyReleased) {
				if (event.key.code == sfKeyLeft) {
					input[BUTTON_LEFT] = 0;
				} else if (event.key.code == sfKeyRight) {
					input[BUTTON_RIGHT] = 0;
				} else if (event.key.code == sfKeyUp || event.key.code == sfKeySpace) {
					input[BUTTON_JUMP] = 0;
				}
			} else if (event.type == sfEvtClosed) {
				sfRenderWindow_close(window);
			} else if (event.type == sfEvtResized) {
				render_scale_window(view, event);
			}
		}

		//Update game state
		game_update(input);
		
		// Update camera
		render_handle_camera(window, view);

		//Clear the screen
		sfRenderWindow_clear(window, sfWhite);

		//Draw background
		render_other(window, view);

		//Draw the tiles and entities
		render_tiles(window, view, draw_overlay);
		render_entities(window, view);

		//Draw coords if needed
		if (draw_overlay) {
			render_overlay(window, view, time);
		}

		//Frametime
		time = sfClock_getElapsedTime(clock);
		// Restart the clock
		sfClock_restart(clock);

		// Update the window
		sfRenderWindow_display(window);
	}

exit:
	// Cleanup resources
	sfRenderWindow_destroy(window);
	sfClock_destroy(clock);
	sfView_destroy(view);
	
	return 0;
}

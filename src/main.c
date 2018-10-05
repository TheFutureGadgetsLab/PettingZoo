#include <SFML/Audio.h>
#include <SFML/Graphics.h>
#include <SFML/Config.h>
#include <SFML/System.h>
#include <stdio.h>
#include <stdlib.h>
#include <game.h>
#include <defs.h>

int rescale_window(sfView *view, sfEvent event);

int main(int argc, char **argv)
{
	sfVideoMode mode = {800, 600, 32};
	sfRenderWindow* window;
	sfVector2f moveby = {0, 0};
	sfEvent event;
	sfTime time;
	sfClock *clock;
	sfView *view;

	// Create the clock
	clock = sfClock_create();

	// Create the main window
	window = sfRenderWindow_create(mode, "SFML window", sfResize | sfClose, NULL);
	if (!window)
		return -1;

	// Vsync 
	sfRenderWindow_setVerticalSyncEnabled(window, sfTrue);

	//Load assets
	game_load_assets();

	//Generate game
	game_gen_map();

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
					moveby.y = -8;
				} else if (event.key.code == sfKeyDown) {
					moveby.y = 8; 
				} else if (event.key.code == sfKeyLeft) {
					moveby.x = -8; 
				}  else if (event.key.code == sfKeyRight) {
					moveby.x = 8; 
				} else if (event.key.code == sfKeyEscape) {
					goto exit;
				}
			} else if (event.type == sfEvtKeyReleased) {
				if (event.key.code == sfKeyUp || event.key.code == sfKeyDown) {
					moveby.y = 0;
				} else if (event.key.code == sfKeyLeft || event.key.code == sfKeyRight) {
					moveby.x = 0; 
				}
			}
		}
		// Get coords
		sfVector2f center = sfView_getCenter(view);
		sfVector2f size = sfView_getSize(view);
		sfVector2f origin;
		origin.x = - center.x + (size.x / 2.0);
		origin.y = - center.y + (size.y / 2.0);

		// Move camera only in correct direction
		if (center.y + moveby.y >= 0 && center.x + moveby.x >= 0 &&
		    center.y + moveby.y < TILE_HEIGHT * LEVEL_HEIGHT && 
			center.x + moveby.x < TILE_WIDTH * LEVEL_WIDTH) {
			sfView_move(view, moveby);
		}
		sfRenderWindow_setView(window, view);

		// Clear the screen
		sfRenderWindow_clear(window, sfBlack);

		//Draw the tiles and entities
		game_draw_tiles(window, view);
		game_draw_entities(window, view);

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
	
	sfView_setSize(view, win_size);

	return 0;
}
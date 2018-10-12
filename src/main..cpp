#include <SFML/Graphics.hpp>
#include <defs.hpp>
#include <rendering.hpp>

int main(int argc, char **argv)
{
	int draw_overlay = 0;
	int input[BUTTON_COUNT] = {0};

    sf::RenderWindow window(sf::VideoMode(800, 600), "PettingZoo");
	sf::Event event;
	sf::Time time;
	sf::Clock clock;
	sf::View view;

	window.setKeyRepeatEnabled(false);

	// Vsync 
	window.setVerticalSyncEnabled(true);

	//Load assets
    // IMPLEMENT!!!!!!!!!!!!! 
	render_load_assets();

	//Generate game
    // IMPLEMENT!!!!!!!!!!!!! 
	// game_setup();

	// Start the game loop
	view = window.getDefaultView();
	while (window.isOpen())
	{
		// Process events
		// while (sfRenderWindow_pollEvent(window, &event))
		while (window.pollEvent(event))
		{
			if (event.type == sf::Event::KeyPressed) {
				if (event.key.code == sf::Keyboard::Up || event.key.code == sf::Keyboard::Space) {
					input[BUTTON_JUMP] = 1;
				} else if (event.key.code == sf::Keyboard::Left) {
					input[BUTTON_LEFT] = 1;
				} else if (event.key.code == sf::Keyboard::Right) {
					input[BUTTON_RIGHT] = 1;
				} else if (event.key.code == sf::Keyboard::Escape) {
					return 0;
				} else if (event.key.code == sf::Keyboard::O) {
					draw_overlay ^= 1;
				}
			} else if (event.type == sf::Event::KeyReleased) {
				if (event.key.code == sf::Keyboard::Left) {
					input[BUTTON_LEFT] = 0;
				} else if (event.key.code == sf::Keyboard::Right) {
					input[BUTTON_RIGHT] = 0;
				} else if (event.key.code == sf::Keyboard::Up || event.key.code == sf::Keyboard::Space) {
					input[BUTTON_JUMP] = 0;
				}
			} else if (event.type == sf::Event::Closed) {
				window.close();
			} else if (event.type == sf::Event::Resized) {
                // IMPLEMENT!!!!!!!!!!!!! 
				// render_scale_window(view, event);
                continue;
			}
		}

		//Update game state
        // IMPLEMENT!!!!!!!!!!!!! 
		// game_update(input);
		
		// Update camera
        // IMPLEMENT!!!!!!!!!!!!! 
		// render_handle_camera(window, view);

		//Clear the screen
		window.clear(sf::Color::White);

		//Draw background
        // IMPLEMENT!!!!!!!!!!!!! 
		// render_other(window, view);

		//Draw the tiles and entities
        // IMPLEMENT!!!!!!!!!!!!! 
		render_tiles(window, view, draw_overlay);
        // IMPLEMENT!!!!!!!!!!!!! 
		// render_entities(window, view);

		//Draw coords if needed
		if (draw_overlay) {
            // IMPLEMENT!!!!!!!!!!!!! 
			// render_overlay(window, view, time);
		}

		//Frametime
		time = clock.getElapsedTime();
		// Restart the clock
		clock.restart();

		// Update the window
		window.display();
	}

	return 0;
}

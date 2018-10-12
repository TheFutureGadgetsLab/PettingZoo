#include <SFML/Graphics.hpp>
#include <defs.hpp>
#include <rendering.hpp>
#include <gamelogic.hpp>
#include <levelgen.hpp>

extern Game game;

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
	window.setVerticalSyncEnabled(true);

	//Load assets
	render_load_assets();

	//Generate game
	game_setup();

	// Start the game loop
	view = window.getDefaultView();
	while (window.isOpen())
	{
		// Process events
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
				render_scale_window(view, event);
                continue;
			}
		}

		//Update game state
		game_update(input);
		
		// Update camera
		render_handle_camera(window, view);

		//Clear the screen
		window.clear(sf::Color::White);

		//Draw background
		render_other(window);

		//Draw the tiles and entities
		render_tiles(window, draw_overlay);
		render_entities(window);

		//Draw coords if needed
		if (draw_overlay) {
			render_overlay(window, time);
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

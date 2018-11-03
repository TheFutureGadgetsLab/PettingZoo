#ifndef GAME_H
#define GAME_H

#include <SFML/Graphics.hpp>
#include <defs.hpp>

class View {
    public:
	sf::Vector2f center;
	sf::Vector2f corner;
	sf::Vector2f size;
};

void render_draw_state(sf::RenderWindow &window);
void render_debug_overlay(sf::RenderWindow &window, sf::Time frametime);
void render_hud(sf::RenderWindow &window, int input[BUTTON_COUNT]);
void render_handle_camera(sf::RenderWindow &window);
void render_load_assets();
void render_scale_window(sf::RenderWindow &window, sf::Event event);
void render_regen_map();

#endif

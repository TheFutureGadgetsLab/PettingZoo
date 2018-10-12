#ifndef GAME_H
#define GAME_H

#include <SFML/Graphics.hpp>

class View {
    public:
	sf::Vector2f center;
	sf::Vector2f corner;
	sf::Vector2f size;
};

void render_tiles(sf::RenderWindow &window, int draw_grid);
void render_entities(sf::RenderWindow &window);
void render_overlay(sf::RenderWindow &window, sf::Time frametime);
void render_other(sf::RenderWindow &window);
void render_handle_camera(sf::RenderWindow &window);
void render_load_assets();
void render_scale_window(sf::RenderWindow &window, sf::Event event);

#endif

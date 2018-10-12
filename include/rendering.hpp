#ifndef GAME_H
#define GAME_H

#include <SFML/Graphics.hpp>

class View {
    public:
	sf::Vector2f center;
	sf::Vector2f corner;
	sf::Vector2f size;
};

void render_tiles(sf::RenderWindow &window, sf::View view, int draw_grid);
void render_entities(sf::RenderWindow window, sf::View view);
void render_overlay(sf::RenderWindow window, sf::View view, sf::Time frametime);
void render_other(sf::RenderWindow window, sf::View view);
void render_handle_camera(sf::RenderWindow window, sf::View view);
void render_load_assets();
void render_scale_window(sf::View view, sf::Event event);

#endif

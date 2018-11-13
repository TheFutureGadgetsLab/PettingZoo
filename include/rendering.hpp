#ifndef RENDERING_H
#define RENDERING_H

#include <SFML/Graphics.hpp>
#include <defs.hpp>

class View {
    public:
	sf::Vector2f center;
	sf::Vector2f corner;
	sf::Vector2f size;
};

void render_draw_state(sf::RenderWindow &window, const struct Game game, const struct Player player);
void render_debug_overlay(sf::RenderWindow &window, const struct Game game, const struct Player player);
void render_hud(sf::RenderWindow &window, const struct Player player, const uint8_t input[BUTTON_COUNT]);
void render_handle_camera(sf::RenderWindow &window, const struct Player player);
void render_load_assets();
void render_scale_window(sf::RenderWindow &window, sf::Event event);
void render_gen_map(const struct Game game);

#endif

#ifndef RENDERING_H
#define RENDERING_H

#include <SFML/Graphics.hpp>
#include <defs.hpp>
#include <gamelogic.hpp>

// Camera parameters
#define CAMERA_INTERP 0.1

class View {
    public:
	sf::Vector2f center;
	sf::Vector2f corner;
	sf::Vector2f size;
};

void render_draw_state(sf::RenderWindow &window, Game &game, Player player);
void render_debug_overlay(sf::RenderWindow &window, Game &game, Player player);
void render_hud(sf::RenderWindow &window, Player player, const uint8_t input[BUTTON_COUNT]);
void render_handle_camera(sf::RenderWindow &window, Player player);
void render_load_assets();
void render_scale_window(sf::RenderWindow &window, sf::Event event);
void render_gen_map(Game& game);

#endif

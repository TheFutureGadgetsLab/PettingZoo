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

void render_draw_state(sf::RenderWindow &window, struct Game *game, struct Player *player);
void render_debug_overlay(sf::RenderWindow &window, struct Game *game, struct Player *player, sf::Time frametime);
void render_hud(sf::RenderWindow &window, struct Player *player, int input[BUTTON_COUNT]);
void render_handle_camera(sf::RenderWindow &window, struct Player *player);
void render_load_assets();
void render_scale_window(sf::RenderWindow &window, sf::Event event);
void render_gen_map(struct Game *game);

#endif

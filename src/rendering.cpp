#include <SFML/Graphics.hpp>
#include <rendering.hpp>
#include <defs.hpp>
#include <string>
#include <gamelogic.hpp>

class GameDisplay {
    public:

};
sf::Sprite sprite_lamp;
sf::Sprite sprite_grid;
sf::Sprite sprite_bg;
sf::Sprite tile_sprites[16];
sf::Text overlay;
sf::Font font;
int tiles_drawn;

View game_view;
extern Game game;

void load_sprite(sf::Sprite sprite, const std::string path, int docenter) {
	sf::Texture tex;
    tex.loadFromFile(path);
	sprite.setTexture(tex, true);
	if (docenter) {
		sf::Vector2u size = tex.getSize();
		sf::Vector2f center = {size.x / 2.0f, size.y / 2.0f};
	    sprite.setOrigin(center);
	}
}

void render_load_assets() {
	// Sprites
	load_sprite(sprite_lamp, "../assets/lamp.png", 0);
	load_sprite(sprite_grid, "../assets/grid.png", 0);
	load_sprite(sprite_bg, "../assets/bg.png", 1);
	load_sprite(tile_sprites[T_GRASS], "../assets/grass.png", 0);
	load_sprite(tile_sprites[T_DIRT], "../assets/dirt.png", 0);
	load_sprite(tile_sprites[T_SPIKES], "../assets/spikes.png", 0);
	load_sprite(tile_sprites[T_BRICKS], "../assets/bricks.png", 0);

	// Text / Font
	font.loadFromFile("../assets/Vera.ttf");
	overlay.setFont(font);
	overlay.setCharacterSize(12);
	overlay.setFillColor(sf::Color::Black);
}

void render_tiles(sf::RenderWindow &window, sf::View view, int draw_grid) {
    printf("Drawing tiles\n");
	int tile_view_x1, tile_view_y1;
	int tile_view_x2, tile_view_y2;
	int x, y;
	sf::Vector2f pos;

	//Calculate bounds for drawing tiles
	tile_view_x1 = game_view.corner.x / TILE_WIDTH;
	tile_view_x2 = (game_view.corner.x + game_view.size.x) / TILE_WIDTH;
	tile_view_y1 = game_view.corner.y / TILE_HEIGHT;
	tile_view_y2 = (game_view.corner.y + game_view.size.y) / TILE_HEIGHT;

	//Loop over tiles and draw them
	int val;
	sf::Sprite sprite;
	tiles_drawn = 0;
	for (x = tile_view_x1; x <= tile_view_x2; x++) {
		if (x >= 0 && x < LEVEL_WIDTH)
		for (y = tile_view_y1; y <= tile_view_y2; y++) {
			if (y < 0 || y >= LEVEL_HEIGHT)
				continue;
			val = game.tiles[y * LEVEL_WIDTH + x];
			if (val > 0) {
				sprite = tile_sprites[val];
				pos.x = x * TILE_WIDTH;
				pos.y = y * TILE_HEIGHT;
			    sprite.setPosition(pos);
				window.draw(sprite);
				tiles_drawn++;
			}

			if (draw_grid) {
				pos.x = x * TILE_WIDTH;
				pos.y = y * TILE_HEIGHT;
				sprite_grid.setPosition(pos);
				window.draw(sprite_grid);
			}
		}
	}
}
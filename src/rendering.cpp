#include <SFML/Graphics.hpp>
#include <rendering.hpp>
#include <defs.hpp>
#include <string>
#include <gamelogic.hpp>
#include <math.h>

sf::Sprite sprites[16];
sf::Texture textures[16];
sf::Text overlay;
sf::Font font;
View game_view;
extern Game game;
extern Player player;
int tiles_drawn;

void load_sprite(int sprite_index, const std::string path, int docenter) {
    textures[sprite_index].loadFromFile(path);
	sprites[sprite_index].setTexture(textures[sprite_index], true);

	if (docenter) {
		sf::Vector2u size = textures[sprite_index].getSize();
		sf::Vector2f center = {size.x / 2.0f, size.y / 2.0f};
	    sprites[sprite_index].setOrigin(center);
	}
}

void update_view_vars(sf::View view) {
	game_view.center = view.getCenter();
	game_view.size = view.getSize();
	game_view.corner.x = game_view.center.x - (game_view.size.x / 2.0);
	game_view.corner.y = game_view.center.y - (game_view.size.y / 2.0);
}

void render_handle_camera(sf::RenderWindow &window) {
	// Candidate camera location, centered on player x position
	sf::View view = window.getDefaultView();
	view.setCenter(player.position_x + 16, LEVEL_PIXEL_HEIGHT - game_view.size.y / 2.0);

	//Set view position and view global variables
	update_view_vars(view);
	if (game_view.corner.x < 0)
		game_view.center.x = game_view.size.x / 2.0;
	if (game_view.corner.x + game_view.size.x > LEVEL_PIXEL_WIDTH)
		game_view.center.x = LEVEL_PIXEL_WIDTH - game_view.size.x / 2.0;

	view.setCenter(game_view.center);
	window.setView(view);
	update_view_vars(view);
}

void render_load_assets() {
	// Sprites
	load_sprite(LAMP, "../assets/lamp.png", 0);
	load_sprite(GRID, "../assets/grid.png", 0);
	load_sprite(BG, "../assets/bg.png", 1);
	load_sprite(GRASS, "../assets/grass.png", 0);
	load_sprite(DIRT, "../assets/dirt.png", 0);
	load_sprite(SPIKES, "../assets/spikes.png", 0);
	load_sprite(BRICKS, "../assets/bricks.png", 0);

	// Text / Font
	font.loadFromFile("../assets/Vera.ttf");
	overlay.setFont(font);
	overlay.setCharacterSize(12);
	overlay.setFillColor(sf::Color::Black);
}

void render_tiles(sf::RenderWindow &window, int draw_grid) {
	int tile_x1, tile_y1;
	int tile_x2, tile_y2;
	int x, y;
	int val;

	//Calculate bounds for drawing tiles
	tile_y1 = game_view.corner.y / TILE_HEIGHT;
	tile_x1 = game_view.corner.x / TILE_WIDTH;
	tile_x2 = (game_view.corner.x + game_view.size.x) / TILE_WIDTH;
	tile_y2 = (game_view.corner.y + game_view.size.y) / TILE_HEIGHT;

	//Loop over tiles and draw them
	tiles_drawn = 0;
	x = (x < 0) ? 0: x;
	for (x = tile_x1; x <= tile_x2 && x < LEVEL_WIDTH; x++) {
		for (y = tile_y1; y <= tile_y2 && y < LEVEL_HEIGHT; y++) {
			if (y < 0)
				continue;
			val = game.tiles[y * LEVEL_WIDTH + x];
			if (val > 0) {
			    sprites[val].setPosition(x * TILE_WIDTH, y * TILE_HEIGHT);
				window.draw(sprites[val]);
				tiles_drawn++;
			}
			
			if (draw_grid) {
				sprites[GRID].setPosition(x * TILE_WIDTH, y * TILE_HEIGHT);
				window.draw(sprites[GRID]);
			}
		}
	}
}

void render_entities(sf::RenderWindow &window) {
	sprites[LAMP].setPosition(player.position_x, player.position_y);
	window.draw(sprites[LAMP]);
}

void render_overlay(sf::RenderWindow &window, sf::Time frametime) {
	char overlay_text[1024];

	sprintf(overlay_text, 
	"Lamp pos: %0.lf, %0.lf\nFPS: %.0lf\nTiles Drawn: %d\nSeed: %u\nVelocity: %.0lf, %0.lf\nTile: %d, %d", 
		player.position_x, player.position_y, 1.0 / frametime.asSeconds(), 
		tiles_drawn, game.seed, player.velocity_x, player.velocity_y,
		player.tile_x, player.tile_y);

	overlay.setString(overlay_text);
	overlay.setPosition(game_view.corner);
	window.draw(overlay);
}

void render_other(sf::RenderWindow &window) {
	sprites[BG].setPosition(game_view.center);
	window.draw(sprites[BG]);
}

void render_scale_window(sf::RenderWindow &window, sf::Event event) {
	int zoom;
	sf::View view = window.getDefaultView();
	sf::Vector2f win_size = {(float)event.size.width, (float)event.size.height};

	//Auto zoom depending on window size
	zoom = round(win_size.y / (LEVEL_PIXEL_HEIGHT));
	zoom = zoom < 1 ? 1 : zoom;
	win_size.x /= zoom;
	win_size.y /= zoom;
	
	view.setSize(win_size);
	update_view_vars(view);
	window.setView(view);
}
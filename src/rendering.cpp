/**
 * @file rendering.hpp
 * @author Haydn Jones, Benjamin Mastripolito
 * @brief Routines for rendering with SMFL
 * @date 2018-12-06
 */

#include <SFML/Graphics.hpp>
#include <rendering.hpp>
#include <defs.hpp>
#include <string>
#include <gamelogic.hpp>
#include <math.h>

class TileMap : public sf::Drawable, public sf::Transformable
{
public:

	bool load_textures(const std::string& tileset) {
        if (!m_tileset.loadFromFile(tileset))
            return false;

		return true;
	}

    bool load_map(const uint8_t *tiles, int width, int height) {
        // resize the vertex array to fit the level size
        m_vertices.setPrimitiveType(sf::Quads);
        m_vertices.resize(width * height * 4);

        // populate the vertex array, with one quad per tile
        for (int i = 0; i < width; ++i)
            for (int j = 0; j < height; ++j)
            {
                // get the current tile number
                int tileNumber = tiles[i + j * width];

                // find its position in the tileset texture
                int tu = tileNumber % (m_tileset.getSize().x / (TILE_SIZE + 2));
                int tv = tileNumber / (m_tileset.getSize().x / (TILE_SIZE + 2));

                // get a pointer to the current tile's quad
                sf::Vertex* quad = &m_vertices[(i + j * width) * 4];

                // define its 4 corners
                quad[0].position = sf::Vector2f(i * TILE_SIZE, j * TILE_SIZE);
                quad[1].position = sf::Vector2f((i + 1) * TILE_SIZE, j * TILE_SIZE);
                quad[2].position = sf::Vector2f((i + 1) * TILE_SIZE, (j + 1) * TILE_SIZE);
                quad[3].position = sf::Vector2f(i * TILE_SIZE, (j + 1) * TILE_SIZE);

                // define its 4 texture coordinates
				// TILE_SIZE + 2 because there is a one pixel halo around each tile
				// and +/- 1 to further account for the halo
                quad[0].texCoords = sf::Vector2f(tu * (TILE_SIZE + 2) + 1, tv * (TILE_SIZE + 2) + 1);
                quad[1].texCoords = sf::Vector2f((tu + 1) * (TILE_SIZE + 2) - 1, tv * (TILE_SIZE + 2) + 1);
                quad[2].texCoords = sf::Vector2f((tu + 1) * (TILE_SIZE + 2) - 1, (tv + 1) * (TILE_SIZE + 2) - 1);
                quad[3].texCoords = sf::Vector2f(tu * (TILE_SIZE + 2) + 1, (tv + 1) * (TILE_SIZE + 2) - 1);
            }

        return true;
    }

private:

    virtual void draw(sf::RenderTarget& target, sf::RenderStates states) const
    {
        // apply the transform
        states.transform *= getTransform();

        // apply the tileset texture
        states.texture = &m_tileset;

        // draw the vertex array
        target.draw(m_vertices, states);
    }

    sf::VertexArray m_vertices;
    sf::Texture m_tileset;
};

sf::Sprite sprites[4];
sf::Texture textures[4];
sf::Text overlay;
sf::Text score;
sf::Font font;
View game_view;
TileMap map;

/**
 * @brief Load a sprite where index is its identifier in sprites array
 * 
 * @param sprite_index The index of the sprite in the sprite collection
 * @param path The path to the image file
 * @param docenter Change the origin of the sprite to be at the center of the image
 */
void load_sprite(int sprite_index, const std::string path, bool docenter = false) {
    textures[sprite_index].loadFromFile(path);
	sprites[sprite_index].setTexture(textures[sprite_index], true);

	if (docenter) {
		sf::Vector2u size = textures[sprite_index].getSize();
		sf::Vector2f center = {size.x / 2.0f, size.y / 2.0f};
	    sprites[sprite_index].setOrigin(center);
	}
}

/**
 * @brief Update the shared view properties
 * 
 * @param view The SFML view object to operate on
 */
void update_view_vars(sf::View view) {
	game_view.center = view.getCenter();
	game_view.size = view.getSize();
	game_view.corner.x = game_view.center.x - (game_view.size.x / 2.0);
	game_view.corner.y = game_view.center.y - (game_view.size.y / 2.0);
}

/**
 * @brief Handle the render camera
 * 
 * @param window The SFML renderwindow object
 * @param player The player object
 */
void render_handle_camera(sf::RenderWindow &window, Player player) {
	// Candidate camera location, centered on player x position
	sf::View view = window.getView();
	sf::Vector2f target;

	//Slide view towards player
	target.x = player.body.px + 16;
	target.y = player.body.py + 16;
	game_view.center.x = game_view.center.x + (target.x - game_view.center.x) * CAMERA_INTERP;
	game_view.center.y = game_view.center.y + (target.y - game_view.center.y) * CAMERA_INTERP;
	view.setCenter(game_view.center);

	//Set view position and view global variables
	update_view_vars(view);
	if (game_view.corner.x < 0)
		game_view.center.x = game_view.size.x / 2.0;
	if (game_view.corner.x + game_view.size.x > LEVEL_PIXEL_WIDTH)
		game_view.center.x = LEVEL_PIXEL_WIDTH - game_view.size.x / 2.0;
	if (game_view.corner.y + game_view.size.y > LEVEL_PIXEL_HEIGHT)
        game_view.center.y = LEVEL_PIXEL_HEIGHT - game_view.size.y / 2.0;

	game_view.center.x = (int)game_view.center.x;
	game_view.center.y = (int)game_view.center.y;
	view.setCenter(game_view.center);
	window.setView(view);
	update_view_vars(view);
}

/**
 * @brief Load visual game assets
 * 
 */
void render_load_assets() {
	// Sprites
	load_sprite(LAMP, "../assets/lamp.png", false);
	load_sprite(BG, "../assets/bg.png", true);
	load_sprite(GRID, "../assets/grid.png", false);

	// Text / Font
	font.loadFromFile("../assets/Vera.ttf");
	overlay.setFont(font);
	overlay.setCharacterSize(12);
	overlay.setFillColor(sf::Color::Black);

	score = overlay;
	score.setString("Score: 00000"); // Set the string so bounds get set properly
	// Centering origin of score text
	sf::FloatRect textRect = score.getLocalBounds();
	score.setOrigin(round(textRect.left + textRect.width/2.0f), round(textRect.top  + textRect.height/2.0f));

	map.load_textures("../assets/spritesheet.png");
}

/**
 * @brief Render the entire map
 * 
 * @param game The game object
 */
void render_gen_map(Game& game) {
	map.load_map(game.tiles, LEVEL_WIDTH, LEVEL_HEIGHT);
}

/**
 * @brief Render all entities (player, enemies)
 * 
 * @param window The SFML window object
 * @param game The game struct
 * @param player The player
 */
void render_entities(sf::RenderWindow &window, Game& game, Player player) {
	sprites[LAMP].setPosition(player.body.px, player.body.py);
	window.draw(sprites[LAMP]);
}

/**
 * @brief Render the debug information overlay
 * 
 * @param window The SFML window object
 * @param game The game struct
 * @param player The player struct
 */
void render_debug_overlay(sf::RenderWindow &window, Game& game, Player player) {
	char overlay_text[512];

	sprintf(overlay_text,
	"Lamp pos: %0.lf, %0.lf\nSeed: %u\nVelocity: %.0lf, %0.lf\nTile: %d, %d",
		player.body.px, player.body.py,
		game.seed, player.body.vx, player.body.vy,
		player.body.tile_x, player.body.tile_y);

	overlay.setString(overlay_text);
	overlay.setPosition(game_view.corner);
	window.draw(overlay);

	int tile_x1, tile_y1;
	int tile_x2, tile_y2;
	int x, y;

	//Calculate bounds for drawing tiles
	tile_x1 = player.body.tile_x - INPUT_SIZE / 2;
	tile_x2 = player.body.tile_x + INPUT_SIZE / 2;
	tile_y1 = player.body.tile_y - INPUT_SIZE / 2;
	tile_y2 = player.body.tile_y + INPUT_SIZE / 2;

	// Bound checking
	tile_x1 = (tile_x1 < 0) ? 0: tile_x1;
	tile_y1 = (tile_y1 < 0) ? 0: tile_y1;
	tile_x2 = (tile_x2 > LEVEL_WIDTH) ? LEVEL_WIDTH: tile_x2;
	tile_y2 = (tile_y2 > LEVEL_HEIGHT) ? LEVEL_HEIGHT: tile_y2;

	//Loop over tiles and draw them
	for (x = tile_x1; x < tile_x2; x++) {
		for (y = tile_y1; y < tile_y2; y++) {	
			sprites[GRID].setPosition(x * TILE_SIZE, y * TILE_SIZE);
			window.draw(sprites[GRID]);
		}
	}
}

/**
 * @brief Render other things
 * 
 * @param window The SFML window object
 */
void render_other(sf::RenderWindow &window) {
	sprites[BG].setPosition(game_view.center);
	window.draw(sprites[BG]);
}

/**
 * @brief Handler for rescaling the viewport
 * 
 * @param window The SFML window object
 * @param event The rescale SFML event
 */
void render_scale_window(sf::RenderWindow &window, sf::Event event) {
	int zoom;
	sf::View view = window.getView();
	sf::Vector2f win_size = {(float)event.size.width, (float)event.size.height};

	//Auto zoom depending on window size
	zoom = round((win_size.x + win_size.y) / (1600 + 900));
	zoom = zoom < 1 ? 1 : zoom;
	win_size.x /= zoom;
	win_size.y /= zoom;

	view.setSize(win_size);
	window.setView(view);
	update_view_vars(view);
}

/**
 * @brief Render the HUD
 * 
 * @param window SFML window object
 * @param player The player structure
 * @param input Player inputs
 */
void render_hud(sf::RenderWindow &window, Player player, const uint8_t input[BUTTON_COUNT]) {
	char score_text[128];

	sprintf(score_text, "Score: %05d\nFitness: %0.2lf\nTime: %0.1lf\n%s %s %s",
	player.score, player.fitness, player.time,
	(input[BUTTON_LEFT] > 0) ? "Left" : "     ",
	(input[BUTTON_RIGHT] > 0) ? "Right" : "     ",
	(input[BUTTON_JUMP] > 0) ? "JUMP" : "");

	score.setString(score_text);
	score.setPosition({game_view.center.x, game_view.corner.y + 10});
	window.draw(score);
}

/**
 * @brief Render everything
 * 
 * @param window SFML window
 * @param game The game struct
 * @param player The player struct
 */
void render_draw_state(sf::RenderWindow &window, Game& game, Player player) {
	render_other(window);
	window.draw(map);
	render_entities(window, game, player);
}
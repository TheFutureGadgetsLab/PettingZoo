#include <SFML/Graphics.h>
#include <stdio.h>
#include <math.h>
#include <defs.h>

struct enemy {
	int init_x;
	int init_y;
	unsigned char type;
};

struct game_obj {
	unsigned char tiles[LEVEL_SIZE];
	struct enemy *enemies[MAX_ENEMIES];
} game;

sfSprite *sprite_tile;
sfSprite *sprite_dirt;
sfSprite *sprite_lamp;
sfSprite *sprite_grid;
sfText *overlay;
sfFont *font;
int tiles_drawn;

void game_gen_map() {
	int x, y, val;

	for (x = 0; x < LEVEL_WIDTH; x++) {
		for (y = 0; y < LEVEL_HEIGHT; y++) {
			val = T_EMPTY;
			if (y == 11)
				val = T_GRASS;
			else if (y > 11)
				val = T_DIRT;
			game.tiles[y * LEVEL_WIDTH + x] = val;
		}
	}
}
void load_sprite(sfSprite **sprite, char *path) {
	sfTexture *tex;
	tex = sfTexture_createFromFile(path, NULL);
	(*sprite) = sfSprite_create();
	sfSprite_setTexture(*sprite, tex, sfTrue);
}

void game_load_assets() {
	// Sprites
	load_sprite(&sprite_tile, "../assets/grass.png");
	load_sprite(&sprite_dirt, "../assets/dirt.png");
	load_sprite(&sprite_lamp, "../assets/lamp.png");
	load_sprite(&sprite_grid, "../assets/grid.png");

	// Text / Font
	font = sfFont_createFromFile("../assets/Vera.ttf");
	overlay = sfText_create();
	sfText_setFont(overlay, font);
	sfText_setCharacterSize(overlay, 12);
	sfText_setColor(overlay, sfBlack);
}

void game_draw_tiles(sfRenderWindow *window, sfView *view, int draw_grid, sfVector2f zoom) {
	int tile_view_x1, tile_view_y1;
	int tile_view_x2, tile_view_y2;
	float view_x, view_y;
	int x, y;
	sfVector2f pos;

	sfVector2f center = sfView_getCenter(view);
	sfVector2f size = sfView_getSize(view);
	view_x = center.x - (size.x / 2.0);
	view_y = center.y - (size.y / 2.0);

	tile_view_x1 = view_x / TILE_WIDTH;
	tile_view_x2 = (view_x + size.x) / TILE_WIDTH;
	tile_view_y1 = view_y / TILE_HEIGHT;
	tile_view_y2 = (view_y + size.y) / TILE_HEIGHT;

	//Loop over tiles and draw them
	int val;
	sfSprite *sprite;
	tiles_drawn = 0;
	for (x = tile_view_x1; x <= tile_view_x2; x++) {
		if (x >= 0 && x < LEVEL_WIDTH)
		//ISSUE: this needs to incorporate zooming
		for (y = tile_view_y1; y <= tile_view_y2; y++) {
			if (y < 0 || y >= LEVEL_HEIGHT)
				continue;
			val = game.tiles[y * LEVEL_WIDTH + x];
			if (val > 0) {
				switch (val) {
					case T_GRASS:
						sprite = sprite_tile;
						break;
					case T_DIRT:
						sprite = sprite_dirt;
						break;
				}
				pos.x = x * TILE_WIDTH;
				pos.y = y * TILE_HEIGHT;
				sfSprite_setPosition(sprite, pos);
				sfRenderWindow_drawSprite(window, sprite, NULL);
				tiles_drawn++;
			}

			if (draw_grid) {
				pos.x = x * TILE_WIDTH;
				pos.y = y * TILE_HEIGHT;
				sfSprite_setPosition(sprite_grid, pos);
				sfRenderWindow_drawSprite(window, sprite_grid, NULL);
			}
		}
	}
}

void game_draw_entities(sfRenderWindow *window, sfView *view) {
	sfVector2f center = sfView_getCenter(view);
	sfSprite_setPosition(sprite_lamp, center);
	sfRenderWindow_drawSprite(window, sprite_lamp, NULL);
}

void game_draw_overlay_text(sfRenderWindow *window, sfView *view, sfTime frametime) {
	char overlay_text[4096];
	sfVector2f lamp_pos = sfSprite_getPosition(sprite_lamp);
	sfVector2f center = sfView_getCenter(view);
	sfVector2f size = sfView_getSize(view);
	sfVector2f origin = {- center.x + (size.x / 2.0), - center.y + (size.y / 2.0)};

	sprintf(overlay_text, "Lamp pos: %0.0lf, %0.0lf\nFPS: %.0lf\nTiles Drawn: %d", 
		lamp_pos.x, lamp_pos.y, 1.0 / sfTime_asSeconds(frametime), tiles_drawn);

	sfText_setString(overlay, overlay_text);
	sfText_setOrigin(overlay, origin);
	sfRenderWindow_drawText(window, overlay, NULL);
}
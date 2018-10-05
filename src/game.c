#include <SFML/Graphics.h>
#include <stdlib.h>
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

sfSprite *sprite_grass;
sfSprite *sprite_dirt;
sfSprite *sprite_bricks;
sfSprite *sprite_lamp;
sfSprite *sprite_grid;

void game_gen_map() {
	int x, y, val;
	srand(3);

	for (x = 0; x < LEVEL_WIDTH; x++) {
		for (y = 0; y < LEVEL_HEIGHT; y++) {
			val = T_EMPTY;
			if (y == 11)
				val = T_GRASS;
			else if (y > 11)
				val = T_DIRT;
			if (y == 12 && rand() % 8 == 0)
				val = T_BRICKS;
			game.tiles[y * LEVEL_WIDTH + x] = val;
		}
	}
}

void load_sprite(sfSprite *sprite, char *path) {
	sfTexture *tex;

	tex = sfTexture_createFromFile(path, NULL);
	sprite = sfSprite_create();
	sfSprite_setTexture(sprite, tex, sfTrue);
}

void game_load_assets() {
	sfTexture *tex;

	load_sprite(sprite_grass, "../assets/grass.png");
	load_sprite(sprite_dirt, "../assets/dirt.png");
	load_sprite(sprite_bricks, "../assets/bricks.png");
	load_sprite(sprite_lamp, "../assets/lamp.png");
	load_sprite(sprite_grid, "../assets/grid.png");
}

void game_draw_tiles(sfRenderWindow *window, sfView *view, int draw_grid) {
	int tile_view_x1, tile_view_y1;
	int tile_view_x2, tile_view_y2;
	int x, y;
	sfVector2f pos;

	sfIntRect vport = sfRenderWindow_getViewport(window, view);
	sfVector2f center = sfView_getCenter(view);
	sfVector2f size = sfView_getSize(view);
	vport.left = center.x - (size.x / 2.0);
	vport.top = center.y - (size.y / 2.0);

	tile_view_x1 = vport.left / TILE_WIDTH;
	tile_view_x2 = (vport.left + vport.width) / TILE_WIDTH;
	tile_view_y1 = vport.top / TILE_HEIGHT;
	tile_view_y2 = (vport.top + vport.height) / TILE_HEIGHT;

	//Loop over tiles and draw them
	int val;
	sfSprite *sprite;
	for (x = tile_view_x1 - 1; x <= tile_view_x2; x++) {
		if (x >= 0 && x < LEVEL_WIDTH)
		//ISSUE: this needs to incorporate zooming
		for (y = tile_view_y1 - 1; y <= tile_view_y2; y++) {
			if (y < 0 || y >= LEVEL_HEIGHT)
				continue;
			val = game.tiles[y * LEVEL_WIDTH + x];
			if (val > 0) {
				switch (val) {
					case T_GRASS:
						sprite = sprite_grass;
						break;
					case T_DIRT:
						sprite = sprite_dirt;
						break;
					case T_BRICKS:
						sprite = sprite_bricks;
						break;
				}
				pos.x = x * TILE_WIDTH;
				pos.y = y * TILE_HEIGHT;
				sfSprite_setPosition(sprite, pos);
				sfRenderWindow_drawSprite(window, sprite, NULL);
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
	char overlay_text[64];
	sfText *overlay = sfText_create();
	sfVector2f lamp_pos = sfSprite_getPosition(sprite_lamp);
	sfVector2f center = sfView_getCenter(view);
	sfVector2f size = sfView_getSize(view);
	sfVector2f origin = {- center.x + (size.x / 2.0), - center.y + (size.y / 2.0)};

	sprintf(overlay_text, "Lamp pos: %0.0lf, %0.0lf", lamp_pos.x, lamp_pos.y);

	sfText_setString(overlay, overlay_text);
	sfText_setOrigin(overlay, origin);
	sfRenderWindow_drawText(window, overlay, NULL);

	origin.y -= 12;
	
	sprintf(overlay_text, "FPS: %.0lf", 1.0 / sfTime_asSeconds(frametime));
	sfText_setString(overlay, overlay_text);
	sfText_setOrigin(overlay, origin);
	sfRenderWindow_drawText(window, overlay, NULL);
}
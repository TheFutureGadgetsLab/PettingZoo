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
sfSprite *sprite_lamp;

void game_gen_map() {
	int x, y, val;

	for (x = 0; x < LEVEL_WIDTH; x++) {
		for (y = 0; y < LEVEL_HEIGHT; y++) {
			val = T_EMPTY;
			if (y > 11)
				val = T_SOLID;
			game.tiles[y * LEVEL_WIDTH + x] = val;
		}
	}
}

void game_load_assets() {
	sfTexture *tile;
	sfTexture *lamp;
	tile = sfTexture_createFromFile("../assets/tile.png", NULL);
	sprite_tile = sfSprite_create();
	sfSprite_setTexture(sprite_tile, tile, sfTrue);

	lamp = sfTexture_createFromFile("../assets/lamp.png", NULL);
	sprite_lamp = sfSprite_create();
	sfVector2f tmp = {2, 2};
	sfSprite_setTexture(sprite_lamp, lamp, sfTrue);
	sfSprite_setScale(sprite_lamp, tmp);
}

void game_draw_tiles(sfRenderWindow *window, sfView *view) {
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
	for (x = tile_view_x1 - 1; x <= tile_view_x2; x++) {
		if (x >= 0 && x < LEVEL_WIDTH)
		for (y = tile_view_y1 - 1; y <= tile_view_y2; y++) {
			if (y < 0 || y >= LEVEL_HEIGHT)
				continue;
			if (game.tiles[y * LEVEL_WIDTH + x] > 0) {
				pos.x = x * TILE_WIDTH;
				pos.y = y * TILE_HEIGHT;
				sfSprite_setPosition(sprite_tile, pos);
				sfRenderWindow_drawSprite(window, sprite_tile, NULL);
			}
		}
	}
}

void game_draw_entities(sfRenderWindow *window, sfView *view) {
	sfVector2f center = sfView_getCenter(view);
	sfSprite_setPosition(sprite_lamp, center);
	sfRenderWindow_drawSprite(window, sprite_lamp, NULL);
}
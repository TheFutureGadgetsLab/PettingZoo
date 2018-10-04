#include <SFML/Graphics.h>
#include <stdio.h>

#define LEVEL_HEIGHT 16
#define LEVEL_WIDTH 64
#define LEVEL_SIZE LEVEL_HEIGHT * LEVEL_WIDTH
#define TILE_WIDTH 64
#define TILE_HEIGHT 64
#define MAX_ENEMIES 32

struct enemy {
	int init_x;
	int init_y;
	unsigned char type;
};

struct game {
	unsigned char tiles[LEVEL_SIZE];
	struct enemy *enemies[MAX_ENEMIES];
};

sfSprite *sprite_tile;

void game_load_assets() {
	sfTexture *tex;
	tex = sfTexture_createFromFile("../assets/tile.png", NULL);
	sprite_tile = sfSprite_create();
	sfSprite_setTexture(sprite_tile, tex, sfTrue);
}

void game_draw_tiles(sfRenderWindow *window) {
	int tile_view_x1, tile_view_y1;
	int tile_view_x2, tile_view_y2;
	int x, y;
	sfVector2f pos;

	//Memory leak?
	sfFloatRect vport = sfView_getViewport(sfRenderWindow_getView(window));

	tile_view_x1 = vport.left / TILE_WIDTH;
	tile_view_x2 = (vport.left + vport.width) / TILE_WIDTH;
	tile_view_y1 = vport.top / TILE_HEIGHT;
	tile_view_y2 = (vport.top + vport.height) / TILE_HEIGHT;

	for (x = tile_view_x1; x < tile_view_x2; x++) {
		for (y = tile_view_y1; y < tile_view_y2; y++) {
			printf("%d,%d ", x, y);
			pos.x = x * TILE_WIDTH;
			pos.y = y * TILE_HEIGHT;
			sfSprite_setPosition(sprite_tile, pos);
			sfRenderWindow_drawSprite(window, sprite_tile, NULL);
		}
	}
}

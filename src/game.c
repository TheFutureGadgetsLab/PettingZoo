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
char overlay_text[32];

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

void game_load_assets() {
	sfTexture *tex;

	// Grass
	tex = sfTexture_createFromFile("../assets/grass.png", NULL);
	sprite_tile = sfSprite_create();
	sfSprite_setTexture(sprite_tile, tex, sfTrue);
	sfVector2u size = sfTexture_getSize(tex);
	sfVector2f scale = {TILE_WIDTH / size.x, TILE_HEIGHT / size.y};
	sfSprite_setScale(sprite_tile, scale);

	// Dirt
	tex = sfTexture_createFromFile("../assets/dirt.png", NULL);
	sprite_dirt = sfSprite_create();
	sfSprite_setTexture(sprite_dirt, tex, sfTrue);
	size = sfTexture_getSize(tex);
	scale.x = TILE_WIDTH / size.x;
	scale.y = TILE_HEIGHT / size.y;
	sfSprite_setScale(sprite_dirt, scale);

	// Lamp
	tex = sfTexture_createFromFile("../assets/lamp.png", NULL);
	sprite_lamp = sfSprite_create();
	sfSprite_setTexture(sprite_lamp, tex, sfTrue);
	size = sfTexture_getSize(tex);
	scale.x = TILE_WIDTH / size.x;
	scale.y = TILE_HEIGHT / size.y;
	sfSprite_setScale(sprite_lamp, scale);

	// Grid
	tex = sfTexture_createFromFile("../assets/grid.png", NULL);
	sprite_grid = sfSprite_create();
	sfSprite_setTexture(sprite_grid, tex, sfTrue);
	size = sfTexture_getSize(tex);
	scale.x = TILE_WIDTH / size.x;
	scale.y = TILE_HEIGHT / size.y;
	sfSprite_setScale(sprite_grid, scale);

	// Text / Font
	font = sfFont_createFromFile("../assets/Vera.ttf");
	overlay = sfText_create();
	sfText_setFont(overlay, font);
	sfText_setCharacterSize(overlay, 12);
	sfText_setColor(overlay, sfBlack);
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
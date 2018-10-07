#include <SFML/Graphics.h>
#include <game.h>
#include <stdio.h>
#include <stdlib.h>
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


struct player_obj player;

sfSprite *sprite_lamp;
sfSprite *sprite_grid;
sfText *overlay;
sfFont *font;
int tiles_drawn;
sfSprite *tile_sprites[16];

void game_gen_map() {
	int x, y, val;
	srand(1234);

	for (x = 0; x < LEVEL_WIDTH; x++) {
		for (y = 0; y < LEVEL_HEIGHT; y++) {
			val = T_EMPTY;
			if (y == 11)
				val = T_GRASS;
			else if (y > 11)
				val = T_DIRT;
			if (y == 10 && rand() % 8 == 0)
				val = T_SPIKES;
			game.tiles[y * LEVEL_WIDTH + x] = val;
		}
	}
}
sfSprite* load_sprite(sfSprite **sprite, char *path, int docenter) {
	sfTexture *tex;
	tex = sfTexture_createFromFile(path, NULL);
	(*sprite) = sfSprite_create();
	sfSprite_setTexture(*sprite, tex, sfTrue);
	if (docenter) {
		sfVector2u size = sfTexture_getSize(tex);
		sfVector2f center = {size.x / 2.0, size.y / 2.0};
		sfSprite_setOrigin(*sprite, center);
	}
	return *sprite;
}
int tile_at(int x, int y) {
	if (x < 0 || x >= LEVEL_WIDTH || y < 0 || y >= LEVEL_HEIGHT)
		return 0;
	return game.tiles[y * LEVEL_WIDTH + x];
}

void game_update() {
	//Player input
	if (player.right)
		player.velocity.x = 6;
	if (player.left)
		player.velocity.x = -6;
	if (player.jump)
		player.velocity.y = -8;
	player.jump = 0;

	//Player physics
	int tile_x = (player.position.x + 16) / TILE_WIDTH;
	int tile_y = (player.position.y + 16) / TILE_HEIGHT;
	int feet_y = (player.position.y + 1) / TILE_HEIGHT + 1;
	int right_x = (player.position.x + 33) / TILE_WIDTH;
	int left_x = (player.position.x - 1) / TILE_WIDTH;
	player.velocity.y += 0.5;
	player.velocity.x /= 1.5;
	if (tile_at(tile_x, feet_y) > 0) {
		if (player.velocity.y > 0)
			player.velocity.y = 0;
		player.position.y = (feet_y - 1) * TILE_HEIGHT;
	}
	if (tile_at(right_x, tile_y) || right_x >= LEVEL_WIDTH) {
		if (player.velocity.x > 0)
			player.velocity.x = 0;
		player.position.x = (right_x - 1) * TILE_WIDTH;
	}
	if (tile_at(left_x, tile_y) || left_x <= 0) {
		if (player.velocity.x < 0)
			player.velocity.x = 0;
		player.position.x = (left_x + 1) * TILE_WIDTH;
	}
	player.position.x += player.velocity.x;
	player.position.y += player.velocity.y;
	if (player.position.y > LEVEL_HEIGHT * TILE_HEIGHT) {
		player.position.y = 0;
		//TODO: Death
	}
}

void game_load_assets() {
	// Sprites
	load_sprite(&sprite_lamp, "../assets/lamp.png", 0);
	load_sprite(&sprite_grid, "../assets/grid.png", 0);
	load_sprite(&tile_sprites[T_GRASS], "../assets/grass.png", 0);
	load_sprite(&tile_sprites[T_DIRT], "../assets/dirt.png", 0);
	load_sprite(&tile_sprites[T_SPIKES], "../assets/spikes.png", 0);
	load_sprite(&tile_sprites[T_BRICKS], "../assets/bricks.png", 0);

	// Text / Font
	font = sfFont_createFromFile("../assets/Vera.ttf");
	overlay = sfText_create();
	sfText_setFont(overlay, font);
	sfText_setCharacterSize(overlay, 12);
	sfText_setColor(overlay, sfBlack);
}

void game_setup() {
	player.position.x = 64;
	player.position.y = 320;
}

void game_draw_tiles(sfRenderWindow *window, sfView *view, int draw_grid) {
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
				sprite = tile_sprites[val];
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
	sfSprite_setPosition(sprite_lamp, player.position);
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
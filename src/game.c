#include <SFML/Graphics.h>
#include <game.h>
#include <stdio.h>
#include <math.h>
#include <defs.h>
#include <time.h>
#include <limits.h>
#include <levelgen.h>

struct game_obj game;
struct player_obj player;
struct view_obj game_view;

sfSprite *sprite_lamp;
sfSprite *sprite_grid;
sfSprite *sprite_bg;
sfText *overlay;
sfFont *font;
sfSprite *tile_sprites[16];
int tiles_drawn;

void load_sprite(sfSprite **sprite, char *path, int docenter) {
	sfTexture *tex;
	tex = sfTexture_createFromFile(path, NULL);
	(*sprite) = sfSprite_create();
	sfSprite_setTexture(*sprite, tex, sfTrue);
	if (docenter) {
		sfVector2u size = sfTexture_getSize(tex);
		sfVector2f center = {size.x / 2.0, size.y / 2.0};
		sfSprite_setOrigin(*sprite, center);
	}
}

int tile_at(int x, int y) {
	if (x < 0 || x >= LEVEL_WIDTH || y < 0 || y >= LEVEL_HEIGHT)
		return 0;
	return game.tiles[y * LEVEL_WIDTH + x];
}

void set_view_vars(sfView *view) {
	game_view.center = sfView_getCenter(view);
	game_view.size = sfView_getSize(view);
	game_view.corner.x = game_view.center.x - (game_view.size.x / 2.0);
	game_view.corner.y = game_view.center.y - (game_view.size.y / 2.0);
}

void game_update(sfRenderWindow *window, sfView *view) {
	// Move camera towards player position
	sfVector2f moveto = {player.position.x + 16, player.position.y + 16};
	sfView_setCenter(view, moveto);

	//Set view position and view global variables
	set_view_vars(view);
	if (game_view.corner.x < 0)
		game_view.center.x = game_view.size.x / 2.0;
	if (game_view.corner.x + game_view.size.x > LEVEL_PIXEL_WIDTH)
		game_view.center.x = LEVEL_PIXEL_WIDTH - game_view.size.x / 2.0;
	if (game_view.corner.y + game_view.size.y > LEVEL_PIXEL_HEIGHT)
		game_view.center.y = LEVEL_PIXEL_HEIGHT - game_view.size.y / 2.0;
	sfView_setCenter(view, game_view.center);
	sfRenderWindow_setView(window, view);
	set_view_vars(view);

	//Player input
	if (player.right)
		player.velocity.x = V_X;
	if (player.left)
		player.velocity.x = -V_X;
	if (player.jump && player.canjump) {
		player.velocity.y = -V_JUMP;
		player.canjump = 0;
	}

	//Player physics
	int tile_x = (player.position.x + 16) / TILE_WIDTH;
	int tile_y = (player.position.y + 16) / TILE_HEIGHT;
	int feet_y = (player.position.y + 33) / TILE_HEIGHT;
	int right_x = (player.position.x + 33) / TILE_WIDTH;
	int left_x = (player.position.x - 1) / TILE_WIDTH;

	//Gravity
	player.velocity.y += GRAVITY;

	//Horizontal inertia
	player.velocity.x /= INTERTA;

	//Collision on bottom
	if (tile_at(tile_x, feet_y) > 0) {
		if (player.velocity.y > 0)
			player.velocity.y = 0;
		player.position.y = (feet_y - 1) * TILE_HEIGHT;
		player.canjump = 1;
	}

	//Right collision
	if (tile_at(right_x, tile_y) || right_x >= LEVEL_WIDTH) {
		if (player.velocity.x > 0)
			player.velocity.x = 0;
		player.position.x = (right_x - 1) * TILE_WIDTH;
	}

	//Left collision
	if (tile_at(left_x, tile_y) || left_x <= 0) {
		if (player.velocity.x < 0)
			player.velocity.x = 0;
		player.position.x = (left_x + 1) * TILE_WIDTH;
	}

	//Apply player velocity
	player.position.x += player.velocity.x;
	player.position.y += player.velocity.y;

	//Lower bound
	if (player.position.y > LEVEL_PIXEL_HEIGHT) {
		player.position.y = 0;
		//TODO: Death
	}
}

void game_load_assets() {
	// Sprites
	load_sprite(&sprite_lamp, "../assets/lamp.png", 0);
	load_sprite(&sprite_grid, "../assets/grid.png", 0);
	load_sprite(&sprite_bg, "../assets/bg.png", 1);
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
	levelgen_gen_map(&game, &game.seed);
	player.position.x = SPAWN_X * TILE_WIDTH;
	player.position.y = SPAWN_Y * TILE_HEIGHT;
}

void game_draw_tiles(sfRenderWindow *window, sfView *view, int draw_grid) {
	int tile_view_x1, tile_view_y1;
	int tile_view_x2, tile_view_y2;
	int x, y;
	sfVector2f pos;

	//Calculate bounds for drawing tiles
	tile_view_x1 = game_view.corner.x / TILE_WIDTH;
	tile_view_x2 = (game_view.corner.x + game_view.size.x) / TILE_WIDTH;
	tile_view_y1 = game_view.corner.y / TILE_HEIGHT;
	tile_view_y2 = (game_view.corner.y + game_view.size.y) / TILE_HEIGHT;

	//Loop over tiles and draw them
	int val;
	sfSprite *sprite;
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

	sprintf(overlay_text, 
	"Lamp pos: %.0lf, %.0lf\nFPS: %.0lf\nTiles Drawn: %d\nSeed: %u\nVelocity: %.0lf, %0.lf", 
		player.position.x, player.position.y, 1.0 / sfTime_asSeconds(frametime), 
		tiles_drawn, game.seed, player.velocity.x, player.velocity.y);

	sfText_setString(overlay, overlay_text);
	sfText_setPosition(overlay, game_view.corner);
	sfRenderWindow_drawText(window, overlay, NULL);
}

void game_draw_other(sfRenderWindow *window, sfView *view) {
	sfSprite_setPosition(sprite_bg, game_view.center);
	sfRenderWindow_drawSprite(window, sprite_bg, NULL);
}
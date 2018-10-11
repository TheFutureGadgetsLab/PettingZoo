#ifndef GAME_H
#define GAME_H

#include <SFML/Graphics.h>
#include <defs.h>

void game_setup();
void game_update(sfRenderWindow *window, sfView *view, int input[BUTTON_COUNT]);
void game_draw_tiles(sfRenderWindow *window, sfView *view, int draw_grid);
void game_draw_entities(sfRenderWindow *window, sfView *view);
void game_draw_overlay_text(sfRenderWindow *window, sfView *view, sfTime frametime);
void game_draw_other(sfRenderWindow *window, sfView *view);
void handle_camera(sfRenderWindow *window, sfView *view);
void game_load_assets();

struct player_obj {
	float position_x;
	float position_y;
	float velocity_x;
	float velocity_y;
	int left;
	int right;
	int jump;
	int canjump;
};

struct view_obj {
	sfVector2f center;
	sfVector2f corner;
	sfVector2f size;
};
extern struct view_obj game_view;

struct enemy {
	int init_x;
	int init_y;
	unsigned char type;
};

struct game_obj {
	unsigned char tiles[LEVEL_SIZE];
	struct enemy *enemies[MAX_ENEMIES];
	unsigned int seed;
};

#endif

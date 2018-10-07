#ifndef GAME_H
#define GAME_H

#include <SFML/Graphics.h>

void game_setup();
void game_update();
void game_draw_tiles(sfRenderWindow *window, sfView *view, int draw_grid);
void game_draw_entities(sfRenderWindow *window, sfView *view);
void game_draw_overlay_text(sfRenderWindow *window, sfView *view, sfTime frametime);
void game_draw_other(sfRenderWindow *window, sfView *view);
void game_gen_map();
void game_load_assets();

struct player_obj {
	sfVector2f position;
	sfVector2f velocity;
	int left;
	int right;
	int jump;
	int canjump;
};

extern struct player_obj player;

#endif

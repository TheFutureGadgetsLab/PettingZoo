#ifndef GAME_H
#define GAME_H

#include <SFML/Graphics.h>

struct view_obj {
	sfVector2f center;
	sfVector2f corner;
	sfVector2f size;
};

void render_tiles(sfRenderWindow *window, sfView *view, int draw_grid);
void render_entities(sfRenderWindow *window, sfView *view);
void render_overlay(sfRenderWindow *window, sfView *view, sfTime frametime);
void render_other(sfRenderWindow *window, sfView *view);
void render_handle_camera(sfRenderWindow *window, sfView *view);
void render_load_assets();
void render_scale_window(sfView *view, sfEvent event);

#endif

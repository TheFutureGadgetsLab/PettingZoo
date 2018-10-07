#ifndef GAME_H
#define GAME_H

#include <SFML/Graphics.h>

void game_draw_tiles(sfRenderWindow *window, sfView *view, int draw_grid);
void game_draw_entities(sfRenderWindow *window, sfView *view);
void game_draw_overlay_text(sfRenderWindow *window, sfView *view, sfTime frametime);
void game_gen_map();
void game_load_assets();
void load_sprite(sfSprite **sprite, char *path);


#endif

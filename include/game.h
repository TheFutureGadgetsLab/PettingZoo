#ifndef GAME_H
#define GAME_H

#include <SFML/Graphics.h>

void game_draw_tiles(sfRenderWindow *window, sfView *view);
void game_gen_map();
void game_load_assets();

#endif

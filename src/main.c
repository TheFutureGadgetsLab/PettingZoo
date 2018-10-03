#include <SFML/Audio.h>
#include <SFML/Graphics.h>
#include <SFML/Graphics/RenderWindow.h>
#include <SFML/Window.h>
#include <SFML/Config.h>
#include <SFML/System/Time.h>
#include <SFML/System/Clock.h>
#include <math.h>

int main()
{
    sfVideoMode mode = {800, 600, 32};
    sfRenderWindow* window;
    sfTexture* texture;
    sfSprite* sprite;
    sfVector2f moveby = {0.1, 0};
    sfEvent event;
    /* Create the main window */
    window = sfRenderWindow_create(mode, "SFML window", sfResize | sfClose, NULL);
    sfWindow_setVerticalSyncEnabled(window, sfTrue);
    if (!window)
        return -1;
    /* Load a sprite to display */
    texture = sfTexture_createFromFile("cute_image.jpg", NULL);
    if (!texture)
        return -1;
    sprite = sfSprite_create();
    sfSprite_setTexture(sprite, texture, sfTrue);
   
    ///* Start the game loop */
    sfClock *clock = sfClock_create();
    while (sfRenderWindow_isOpen(window))
    {
        /* Process events */
        while (sfRenderWindow_pollEvent(window, &event))
        {
            /* Close window : exit */
            if (event.type == sfEvtClosed)
                sfRenderWindow_close(window);
        }
        /* Clear the screen */
        sfRenderWindow_clear(window, sfBlack);
        /* Draw the sprite */
        sfTime time = sfClock_getElapsedTime(clock);
        moveby.x = sinf(sfTime_asSeconds(time) * 4.0) * 2.0;
        sfSprite_move(sprite, moveby);
        sfRenderWindow_drawSprite(window, sprite, NULL);
        /* Draw the text */
        //sfRenderWindow_drawText(window, text, NULL);
        /* Update the window */
        sfRenderWindow_display(window);
    }
    /* Cleanup resources */
    //sfMusic_destroy(music);
    //sfText_destroy(text);
    //sfFont_destroy(font);
    sfSprite_destroy(sprite);
    sfTexture_destroy(texture);
    sfRenderWindow_destroy(window);
    return 0;
}

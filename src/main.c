#include <SFML/Audio.h>
#include <SFML/Graphics.h>
#include <SFML/Window.h>
#include <SFML/Config.h>
#include <SFML/System.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    sfVideoMode mode = {800, 800, 32};
    sfRenderWindow* window;
    sfTexture* texture;
    sfSprite* sprite;
    sfVector2f moveby = {0, 0};
    sfEvent event;
    sfTime time;
    sfFont *font;
    sfText *text;

    char framerate_txt[32];
   
    // Create the main window
    window = sfRenderWindow_create(mode, "SFML window", sfResize | sfClose, NULL);
    if (!window)
        return -1;

    // Vsync 
    sfRenderWindow_setVerticalSyncEnabled(window, sfTrue);

    // Load a sprite to display
    texture = sfTexture_createFromFile("../assets/cute_image.jpg", NULL);
    if (!texture)
        return -1;

    // Framerate text
    font = sfFont_createFromFile("../assets/Rajdhani-Regular.ttf");
    if (!font)
        return EXIT_FAILURE;
    text = sfText_create();
    sfText_setFont(text, font);
    sfText_setCharacterSize(text, 50);
    sprite = sfSprite_create();
    sfSprite_setTexture(sprite, texture, sfTrue);
   
    // Start the game loop
    sfClock *clock = sfClock_create();
    sfClock *frame_clock = sfClock_create();
    while (sfRenderWindow_isOpen(window))
    {
        // Process events
        while (sfRenderWindow_pollEvent(window, &event))
        {
            // Close window
            if (event.type == sfEvtClosed) {
                printf("Exiting\n");
                sfRenderWindow_close(window);
            } else if (event.type == sfEvtKeyPressed) {
                if (event.key.code == sfKeyUp) {
                    printf("Up\n");
                    moveby.y = -1.5; 
                } else if (event.key.code == sfKeyDown) {
                    printf("Down\n");
                    moveby.y = 1.5; 
                } else if (event.key.code == sfKeyLeft) {
                    printf("Left\n");
                    moveby.x = -1.5; 
                }  else if (event.key.code == sfKeyRight) {
                    printf("Right\n");
                    moveby.x = 1.5; 
                }
            } else if (event.type == sfEvtKeyReleased) {
                if (event.key.code == sfKeyUp || event.key.code == sfKeyDown) {
                    moveby.y = 0;
                } else if (event.key.code == sfKeyLeft || event.key.code == sfKeyRight) {
                    moveby.x = 0; 
                }
            }
        }

        // Clear the screen
        sfRenderWindow_clear(window, sfBlack);

        // Get framerate
        sfTime frametime = sfClock_getElapsedTime(frame_clock);
        sprintf(framerate_txt, "%.0lf", 1.0 / sfTime_asSeconds(frametime));
        sfText_setString(text, framerate_txt);

        // Draw the sprite
        // time = sfClock_getElapsedTime(clock);
        sfSprite_move(sprite, moveby);
        sfRenderWindow_drawSprite(window, sprite, NULL);

        // Draw the text 
        sfRenderWindow_drawText(window, text, NULL);
        
        // Restart frame clock
        sfClock_restart(frame_clock);

        // Update the window
        sfRenderWindow_display(window);
    }

    // Cleanup resources
    //sfMusic_destroy(music);
    sfText_destroy(text);
    sfFont_destroy(font);
    sfSprite_destroy(sprite);
    sfTexture_destroy(texture);
    sfRenderWindow_destroy(window);
    
    return 0;
}

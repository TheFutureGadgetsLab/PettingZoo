all: example

example: example.c
	gcc -g -Wall -lcsfml-graphics -lm -lcsfml-system -lcsfml-window -lcsfml-audio example.c
clean:
	rm -rf a.out
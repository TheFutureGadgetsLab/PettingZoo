all: example

example: example.c
	gcc -g -Wall -lcsfml-graphics -lcsfml-audio example.c
clean:
	rm -rf a.out
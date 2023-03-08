LUA_INC=-I /usr/local/include
LUA_LIB=-L /usr/local/bin -llua54
CFLAGS=-Wall -O2
SHARED=--shared
SO=dll

all : mnist.$(SO) ann.$(SO)

mnist.$(SO) : mnist.c
	gcc -o $@ $(SHARED) $(CFLAGS) $^ $(LUA_INC) $(LUA_LIB)

ann.$(SO) : ann.c
	gcc -o $@ $(SHARED) $(CFLAGS) $^ $(LUA_INC) $(LUA_LIB)

clean :
	rm -f *.$(SO)

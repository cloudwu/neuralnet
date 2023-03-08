#define LUA_LIB
#include <lua.h>
#include <lauxlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

static int
label_get(lua_State *L) {
	uint8_t *data = (uint8_t *)luaL_checkudata(L, 1, "MNIST_LABELS");
	int n = luaL_checkinteger(L, 2);
	int sz = lua_rawlen(L, 1);
	if (n <= 0 || n > sz) {
		return luaL_error(L, "Out of range %d [1, %d]", n, sz);
	}
	lua_pushinteger(L, data[n-1]);
	return 1;
}

static int
label_len(lua_State *L) {
	luaL_checkudata(L, 1, "MNIST_LABELS");
	int sz = lua_rawlen(L, 1);
	lua_pushinteger(L, sz);
	return 1;
}

static uint32_t
read_uint32(FILE *f) {
	uint8_t bytes[4] = {0} ;
	fread(bytes, 1, 4, f);
	return bytes[0] << 24 | bytes[1] << 16 | bytes[2] << 8 | bytes[3];
}

static int
read_labels(lua_State *L) {
	const char * filename = luaL_checkstring(L, 1);
	FILE *f = fopen(filename, "rb");
	if (f == NULL)
		return luaL_error(L, "Can't open %s", filename);
	uint32_t magic = read_uint32(f);
	if (magic != 2049)
		return luaL_error(L, "Invalid magic number %d (Should be 2049)", magic);
	uint32_t number = read_uint32(f);
	void *data = lua_newuserdatauv(L, number, 0);
	if (fread(data, 1, number, f) != number)
		return luaL_error(L, "Invalid labels number (%d)", number);
	fclose(f);
	if (luaL_newmetatable(L, "MNIST_LABELS")) {
		luaL_Reg l[] = {
			{ "__index", label_get },
			{ "__len", label_len },
			{ NULL, NULL },
		};
		luaL_setfuncs(L, l, 0);
	}
	lua_setmetatable(L, -2);
	return 1;
}

struct image_meta {
	uint32_t n;
	uint32_t row;
	uint32_t col;
};

static int
image_len(lua_State *L) {
	luaL_checkudata(L, 1, "MNIST_IMAGES");
	lua_getiuservalue(L, 1, 1);
	struct image_meta *meta = (struct image_meta *)lua_touserdata(L, -1);
	lua_pushinteger(L, meta->n);
	return 1;
}

static int
image_attrib(lua_State *L, struct image_meta *meta, const char *what) {
	if (strcmp(what, "row") == 0) {
		lua_pushinteger(L, meta->row);
		return 1;
	} else if (strcmp(what, "col") == 0) {
		lua_pushinteger(L, meta->col);
		return 1;
	}
	return luaL_error(L, "Can't get .%s", what);
}

static int
image_get(lua_State *L) {
	luaL_checkudata(L, 1, "MNIST_IMAGES");
	lua_getiuservalue(L, 1, 1);
	struct image_meta *meta = (struct image_meta *)lua_touserdata(L, -1);
	if (lua_type(L, 2) == LUA_TSTRING) {
		return image_attrib(L, meta, lua_tostring(L, 2));
	}
	int idx = luaL_checkinteger(L, 2);
	if (idx <= 0 || idx > meta->n) {
		return luaL_error(L, "Out of range %d [1, %d]", idx, meta->n);
	}
	size_t stride = meta->row * meta->col;
	const char * image = (const char *)lua_touserdata(L, 1);
	image = image + stride * (idx-1);
	lua_pushlstring(L, image, stride);
	return 1;
}

static int
read_images(lua_State *L) {
	struct image_meta *meta = (struct image_meta *)lua_newuserdatauv(L, sizeof(*meta), 0);
	const char * filename = luaL_checkstring(L, 1);
	FILE *f = fopen(filename, "rb");
	if (f == NULL)
		return luaL_error(L, "Can't open %s", filename);
	uint32_t magic = read_uint32(f);
	if (magic != 2051)
		return luaL_error(L, "Invalid magic number %d (Should be 2051)", magic);
	meta->n = read_uint32(f);
	meta->row = read_uint32(f);
	meta->col = read_uint32(f);
	size_t sz = meta->n * meta->row * meta->col;
	void *data = lua_newuserdatauv(L, sz, 1);
	lua_pushvalue(L, -2);
	lua_setiuservalue(L, -2, 1);
	if (fread(data, 1, sz, f) != sz)
		return luaL_error(L, "Invalid images size %dx%dx%d", meta->n, meta->row, meta->col);
	fclose(f);
	if (luaL_newmetatable(L, "MNIST_IMAGES")) {
		luaL_Reg l[] = {
			{ "__index", image_get },
			{ "__len", image_len },
			{ NULL, NULL },
		};
		luaL_setfuncs(L, l, 0);
	}
	lua_setmetatable(L, -2);
	return 1;
}

static int
gen_pgm(lua_State *L) {
	size_t sz = 0;
	const uint8_t * image = (const uint8_t *)luaL_checklstring(L, 1, &sz);
	int row = luaL_checkinteger(L, 2);
	int col = luaL_checkinteger(L, 3);
	size_t stride = row * col;
	if (stride != sz)
		return luaL_error(L, "Invalid %d x %d", row, col);
	luaL_Buffer b;
	luaL_buffinit(L, &b);
	lua_pushfstring(L, "P5\n%d %d\n255\n", row, col);
	luaL_addvalue(&b);
	char * buffer = luaL_prepbuffsize(&b, stride);
	memcpy(buffer, image, stride);
	luaL_addsize(&b, stride);
	luaL_pushresult(&b);
	return 1;
}

LUAMOD_API int
luaopen_mnist(lua_State *L) {
	luaL_checkversion(L);
	luaL_Reg l[] = {
		{ "labels", read_labels },
		{ "images", read_images },
		{ "pgm", gen_pgm },
		{ NULL, NULL },
	};
	luaL_newlib(L, l);
	return 1;
}

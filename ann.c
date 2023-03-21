#define LUA_LIB

#include <lua.h>
#include <lauxlib.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

struct signal {
	int n;
	float data[1];
};

static inline struct signal *
check_signal(lua_State *L, int index) {
	return (struct signal *)luaL_checkudata(L, index, "ANN_SIGNAL");
}

static int
lsignal_toarray(lua_State *L) {
	struct signal * s = check_signal(L, 1);
	lua_createtable(L, s->n, 0);
	int i;
	for (i=0;i<s->n;i++) {
		lua_pushnumber(L, s->data[i]);
		lua_rawseti(L, -2, i+1);
	}
	return 1;
}

static int
lsignal_size(lua_State *L) {
	struct signal * s = check_signal(L, 1);
	lua_pushinteger(L, s->n);
	return 1;
}

static void
init_signal_with_string(lua_State *L, struct signal *s, int index) {
	size_t sz;
	const uint8_t * image = (const uint8_t *)luaL_checklstring(L, index, &sz);
	if (sz != s->n)
		luaL_error(L, "Invalid image size %d != %d", (int)sz, s->n);
	int i;
	for (i=0;i<s->n;i++) {
		s->data[i] = image[i] / 255.0f;
	}
}

static void
init_signal_with_table(lua_State *L, struct signal *s, int index) {
	int i;
	for (i=0;i<s->n;i++) {
		if (lua_geti(L, index, i+1) != LUA_TNUMBER)
			luaL_error(L, "Invalid signal init %d", i+1);
		s->data[i] = lua_tonumber(L, -1);
		lua_pop(L, 1);
	}
	if (lua_geti(L, index, i+1) != LUA_TNIL)
		luaL_error(L, "Invalid signal init table (too long)");
	lua_pop(L, 1);
}

static void
init_signal_n(lua_State *L, struct signal *s, int n) {
	if (n < 0 || n >= s->n)
		luaL_error(L, "Invalid n (%d)", n);
	memset(s->data, 0, sizeof(s->data[0]) * s->n);
	s->data[n] = 1.0f;
}

static int
lsignal_init(lua_State *L) {
	struct signal * s = check_signal(L, 1);
	switch (lua_type(L, 2)) {
	case LUA_TSTRING:
		init_signal_with_string(L, s, 2);
		break;
	case LUA_TNUMBER:
		init_signal_n(L, s, luaL_checkinteger(L, 2));
		break;
	case LUA_TTABLE:
		init_signal_with_table(L, s, 2);
		break;
	case LUA_TNIL:
	case LUA_TNONE:
		memset(s->data, 0, sizeof(s->data[0]) * s->n);
		break;
	default:
		return luaL_argerror(L, 2, "Invalid signal init arg");
	}
	lua_settop(L, 1);
	return 1;
}

static int
lsignal_max(lua_State *L) {
	struct signal * s = check_signal(L, 1);
	float m = s->data[0];
	float sum = m;
	int idx = 0;
	int i;
	for (i=1;i<s->n;i++) {
		if (s->data[i] > m) {
			m = s->data[i];
			idx = i;
		}
		sum += s->data[i];
	}
	lua_pushinteger(L, idx);
	lua_pushnumber(L, m / sum);
	return 2;
}

static int
lsignal_accumulate(lua_State *L) {
	struct signal * s = check_signal(L, 1);
	struct signal * delta = check_signal(L, 2);
	if (s->n != delta->n)
		return luaL_error(L, "signal size %d != %d", s->n, delta->n);
	int i;
	if (lua_type(L, 3) == LUA_TNUMBER) {
		float eta = lua_tonumber(L, 3);
		for (i=0;i<s->n;i++) {
			s->data[i] += delta->data[i] * eta;
		}
	} else {
		for (i=0;i<s->n;i++) {
			s->data[i] += delta->data[i];
		}
	}
	lua_settop(L, 1);
	return 1;
}

static inline float 
sigmoid(float z) {
	return 1.0f / (1.0f + expf(-z));
}

static int
lsignal_sigmoid(lua_State *L) {
	struct signal * s = check_signal(L, 1);
	int i;
	for (i=0;i<s->n;i++) {
		s->data[i] = sigmoid(s->data[i]);
	}
	lua_settop(L, 1);
	return 1;
}

static inline float
sigmoid_prime(float s) {
	return s * (1-s);
}

static void
addfloat(lua_State *L, luaL_Buffer *b, float f) {
	char tmp[16];
	int n = snprintf(tmp+1, sizeof(tmp)-1, "%.5g", f);
	int k;
	for (k=n+1;k<sizeof(tmp);k++) {
		tmp[k] = ' ';
	}
	if (tmp[1] == '-') {
		lua_pushlstring(L, tmp+1, sizeof(tmp)-1);
	} else {
		tmp[0] = ' ';
		lua_pushlstring(L, tmp, sizeof(tmp)-1);
	}
	luaL_addvalue(b);
}

static int
lsignal_dump(lua_State *L) {
	struct signal * s = check_signal(L, 1);
	int i;
	luaL_Buffer b;
	luaL_buffinit(L, &b);

	luaL_addchar(&b, '[');
	for (i=0;i<s->n;i++) {
		addfloat(L, &b, s->data[i]);
	}
	luaL_addchar(&b, ']');
	luaL_pushresult(&b);
	return 1;
}


static inline void
gaussrand(float r[2], float deviation) {
	float V1, V2, S;
	do {
		float U1 = (double)rand() / RAND_MAX;
		float U2 = (double)rand() / RAND_MAX;

		V1 = 2 * U1 - 1;
		V2 = 2 * U2 - 1;
		S = V1 * V1 + V2 * V2;
	} while (S >= 1 || S == 0);

	float X = sqrtf(-2 * logf(S) / S) * deviation;
	r[0] = V1 * X;
	r[1] = V2 * X;
}

static void
randn(float *f, int n, float deviation) {
	int i;
	for (i=0;i<n;i+=2) {
		gaussrand(f+i, deviation);
	}
	if (n & 1) {
		float tmp[2];
		gaussrand(tmp, deviation);
		f[n-1] = tmp[0];
	}
}

static int
lsignal_randn(lua_State *L) {
	struct signal *s = check_signal(L, 1);
	float deviation = luaL_optnumber(L, 2, 1.0f);
	randn(s->data, s->n, deviation);
	lua_settop(L, 1);
	return 1;
}

static int
lsignal(lua_State *L) {
	int n = luaL_checkinteger(L, 1);
	size_t sz = sizeof(struct signal) + sizeof(float) * (n-1);
	struct signal * s = (struct signal *)lua_newuserdatauv(L, sz, 0);
	memset(s->data, 0, sizeof(s->data[0]) * n);
	s->n = n;
	if (luaL_newmetatable(L, "ANN_SIGNAL")) {
		lua_pushvalue(L, -1);
		lua_setfield(L, -2, "__index");
		luaL_Reg l[] = {
			{ "toarray", lsignal_toarray },
			{ "init", lsignal_init },
			{ "randn", lsignal_randn },
			{ "max", lsignal_max },
			{ "size", lsignal_size },
			{ "accumulate", lsignal_accumulate },
			{ "sigmoid", lsignal_sigmoid },
			{ "__tostring", lsignal_dump },
			{ NULL, NULL },
		};
		luaL_setfuncs(L, l, 0);
	}
	lua_setmetatable(L, -2);

	return 1;
}

struct weight {
	int w;
	int h;
	float data[1];
};

static inline struct weight *
check_weight(lua_State *L, int index) {
	return (struct weight *)luaL_checkudata(L, index, "ANN_WEIGHT");
}

static int
lweight_zero(lua_State *L) {
	struct weight *w = check_weight(L, 1);
	int s = w->w * w->h;
	memset(w->data, 0, sizeof(w->data[0]) * s);
	lua_settop(L, 1);
	return 1;
}

static int
lweight_size(lua_State *L) {
	struct weight *w = check_weight(L, 1);
	lua_pushinteger(L, w->w);
	lua_pushinteger(L, w->h);
	return 2;
}

static int
lweight_randn(lua_State *L) {
	struct weight *w = check_weight(L, 1);
	float deviation = luaL_optnumber(L, 2, 1.0f);
	randn(w->data, w->w * w->h, deviation);
	lua_settop(L, 1);
	return 1;
}

static int
lweight_dump(lua_State *L) {
	struct weight *w = check_weight(L, 1);
	luaL_Buffer b;
	luaL_buffinit(L, &b);
	luaL_addchar(&b, '[');
	int i,j;
	const float * f = w->data;
	for (i=0;i<w->h;i++) {
		luaL_addlstring(&b, "[ ", 2);
		for (j=0;j<w->w;j++) {
			addfloat(L, &b, *f);
			++f;
		}
		luaL_addchar(&b, ']');
		if (i<w->h-1) {
			luaL_addlstring(&b, "\n ", 2);
		}
	}
	luaL_addchar(&b, ']');
	luaL_pushresult(&b);
	return 1;
}

static int
lweight_accumulate(lua_State *L) {
	struct weight * s = check_weight(L, 1);
	struct weight * delta = check_weight(L, 2);
	if (s->w != delta->w || s->h != delta->h)
		return luaL_error(L, "weight size (%d, %d) != (%d, %d)", s->w, s->h, delta->w, delta->h);
	int i;
	int sz = s->w * s->h;
	if (lua_type(L, 3) == LUA_TNUMBER) {
		float eta = lua_tonumber(L, 3);
		for (i=0;i<sz;i++) {
			s->data[i] += delta->data[i] * eta;
		}
	} else {
		for (i=0;i<sz;i++) {
			s->data[i] += delta->data[i];
		}
	}
	lua_settop(L, 1);
	return 1;
}

static int
lweight_import(lua_State *L) {
	struct weight * w = check_weight(L, 1);
	luaL_checktype(L, 2, LUA_TTABLE);
	int i,j;
	float *data = w->data;
	for (i=0;i<w->h;i++) {
		if (lua_geti(L, 2, i+1) != LUA_TTABLE) {
			return luaL_error(L, "Invalid source [%d]", i+1);
		}
		for (j=0;j<w->w;j++) {
			if (lua_geti(L, -1, j+1) != LUA_TNUMBER) {
				return luaL_error(L, "Invalid source [%d][%d]", i+1, j+1);
			}
			*data = lua_tonumber(L, -1);
			lua_pop(L, 1);
			++data;
		}
		if (lua_geti(L, -1, w->w+1) != LUA_TNIL)
			return luaL_error(L, "Invalid source [%d] (too long)", i+1);
		lua_pop(L, 2);
	}
	return 0;
}

static int
lweight(lua_State *L) {
	int width = luaL_checkinteger(L, 1);
	int height = luaL_checkinteger(L, 2);
	int s = width * height;
	size_t sz = sizeof(struct weight) + sizeof(float) * (s-1);
	struct weight * w = (struct weight *)lua_newuserdatauv(L, sz, 0);
	w->w = width;
	w->h = height;
	if (luaL_newmetatable(L, "ANN_WEIGHT")) {
		lua_pushvalue(L, -1);
		lua_setfield(L, -2, "__index");
		luaL_Reg l[] = {
			{ "import", lweight_import },
			{ "zero", lweight_zero },
			{ "randn", lweight_randn },
			{ "size", lweight_size },
			{ "accumulate", lweight_accumulate },
			{ "__tostring", lweight_dump },
			{ NULL, NULL },
		};
		luaL_setfuncs(L, l, 0);
	}
	lua_setmetatable(L, -2);
	return 1;
}

static int
lprop(lua_State *L) {
	struct signal * input = check_signal(L, 1);
	struct signal * output = check_signal(L, 2);
	struct weight * w = check_weight(L, 3);
	if (input->n != w->w || output->n != w->h) {
		return luaL_error(L, "Invalid weight (%d , %d) != (%d , %d)", w->w, w->h, input->n, output->n);
	}
	int i,j;
	const float * c = w->data;
	for (i=0;i<output->n;i++) {
		float s = 0;
		for (j=0;j<input->n;j++) {
			s += input->data[j] * (*c);
			++c;
		}
		output->data[i] = s;
	}
	return 0;
}

// source(w) <----w(w,h)---- delta(h)

static int
lbackprop_weight(lua_State *L) {
	struct signal * source = check_signal(L, 1);
	struct signal * delta = check_signal(L, 2);
	struct weight * w = check_weight(L, 3);
	if (source->n != w->w || delta->n != w->h) {
		return luaL_error(L, "Invalid weight (%d , %d) != (%d, %d)", w->w, w->h, source->n, delta->n);
	}
	int i,j;
	float * nabla = w->data;
	for (i=0;i<delta->n;i++) {
		float d = delta->data[i];
		for (j=0;j<source->n;j++) {
			*nabla = d * source->data[j];
			++nabla;
		}
	}
	return 0;
}


// output_delta(w) <----w(w,h)----- delta(h)

static int
lbackprop_bias(lua_State *L) {
	struct signal * output = check_signal(L, 1);
	struct signal * delta = check_signal(L, 2);
	struct weight * w = check_weight(L, 3);
	if (output->n != w->w || delta->n != w->h) {
		return luaL_error(L, "Invalid weight (%d , %d) != (%d, %d)", w->w, w->h, output->n, delta->n);
	}
	int i,j;
	for (i=0;i<output->n;i++) {
		const float * weight = &w->data[i];
		float d = 0;
		for (j=0;j<delta->n;j++) {
			d += delta->data[j] * (*weight);
			weight += w->w;
		}
		output->data[i] = d;
	}
	return 0;
}

static int
lsigmoid_prime(lua_State *L) {
	struct signal * s = check_signal(L, 1);
	struct signal * mul = check_signal(L, 2);
	struct signal * output = check_signal(L, 3);
	if (s->n != mul->n || s->n != output->n)
		return luaL_error(L, "Invalid signal size");
	int i;
	for (i=0;i<s->n;i++) {
		output->data[i] = sigmoid_prime(s->data[i]) * mul->data[i];
	}
	return 0;
}

static int
lsignal_error(lua_State *L) {
	struct signal * a = check_signal(L, 1);
	struct signal * b = check_signal(L, 2);
	struct signal * output = check_signal(L, 3);
	if (a->n != b->n || a->n != output->n)
		return luaL_error(L, "Invalid signal size");
	int i;
	for (i=0;i<a->n;i++) {
		output->data[i] = a->data[i] - b->data[i];
	}
	return 0;
}

LUAMOD_API int
luaopen_ann(lua_State *L) {
	luaL_checkversion(L);
	luaL_Reg l[] = {
		{ "signal" , lsignal },
		{ "weight", lweight },
		{ "prop", lprop },
		{ "backprop_weight", lbackprop_weight },
		{ "backprop_bias", lbackprop_bias },
		{ "signal_error", lsignal_error },
		{ "sigmoid_prime", lsigmoid_prime },
		{ NULL, NULL },
	};
	luaL_newlib(L, l);
	return 1;
}

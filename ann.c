#define LUA_LIB

#include <lua.h>
#include <lauxlib.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

static int
llayer_toarray(lua_State *L) {
	float * f = (float *)luaL_checkudata(L, 1, "ANN_LAYER");
	int n = lua_rawlen(L, 1) / sizeof(float);
	lua_createtable(L, n, 0);
	int i;
	for (i=0;i<n;i++) {
		lua_pushnumber(L, f[i]);
		lua_rawseti(L, -2, i+1);
	}
	return 1;
}

struct layer {
	int n;
	float *data;
};

static struct layer
checklayer(lua_State *L, int idx) {
	struct layer layer;
	layer.data = luaL_checkudata(L, idx, "ANN_LAYER");
	layer.n = lua_rawlen(L, idx) / sizeof(float);
	return layer;
}

static int
llayer_init(lua_State *L) {
	struct layer layer = checklayer(L, 1);
	size_t sz = 0;
	const uint8_t * image = (const uint8_t *)luaL_checklstring(L, 2, &sz);
	if (sz != layer.n)
		return luaL_error(L, "Invalid image size %d != %d", (int)sz, layer.n);
	int i;
	for (i=0;i<layer.n;i++) {
		layer.data[i] = image[i] / 255.0f;
	}
	lua_settop(L, 1);
	return 1;
}

static int
llayer_init_n(lua_State *L) {
	struct layer layer = checklayer(L, 1);
	int n = luaL_checkinteger(L, 2);
	if (n < 0 || n >= layer.n)
		return luaL_error(L, "Invalid n (%d)", n);
	memset(layer.data, 0, sizeof(float) * layer.n);
	layer.data[n] = 1.0f;
	lua_settop(L, 1);
	return 1;
}

static int
llayer_max(lua_State *L) {
	struct layer layer = checklayer(L, 1);
	float m = layer.data[0];
	float s = m;
	int idx = 0;
	int i;
	for (i=1;i<layer.n;i++) {
		if (layer.data[i] > m) {
			m = layer.data[i];
			idx = i;
		}
		s += layer.data[i];
	}
	lua_pushinteger(L, idx);
	lua_pushnumber(L, m / s);
	return 2;
}

static int
llayer(lua_State *L) {
	int n = luaL_checkinteger(L, 1);
	float * f = (float *)lua_newuserdatauv(L, n * sizeof(*f), 0);
	memset(f, 0, sizeof(*f) * n);
	if (luaL_newmetatable(L, "ANN_LAYER")) {
		lua_pushvalue(L, -1);
		lua_setfield(L, -2, "__index");
		luaL_Reg l[] = {
			{ "toarray", llayer_toarray },
			{ "init", llayer_init },
			{ "init_n", llayer_init_n },
			{ "max", llayer_max },
			{ NULL, NULL },
		};
		luaL_setfuncs(L, l, 0);
	}
	lua_setmetatable(L, -2);
	return 1;
}

struct connection {
	int input_n;
	int output_n;
};

static inline float
randf() {
	float f = ((rand() & 0x7fff) + 1) / (float)0x8000;
	return f;
}

static inline float
randnorm(float r1, float r2) {
	static const float PI = 3.1415927f;
	float x = sqrtf( -2.0 * logf ( r1 ) ) * cosf ( 2.0 * PI * r2 );
	return x;
}

static int
lconnection_randn(lua_State *L) {
	float *f = (float *)luaL_checkudata(L, 1, "ANN_CONNECTION");
	int n = lua_rawlen(L, 1) / sizeof(float);
	float s = randf();
	int i;
	for (i=0;i<n;i++) {
		float r = randf();
		f[i] = randnorm(s, r);
		s = r;
	}
	return 0;
}

static int
lconnection_accumulate(lua_State *L) {
	float *base = (float *)luaL_checkudata(L, 1, "ANN_CONNECTION");
	int base_n = lua_rawlen(L, 1) / sizeof(float);
	float *delta = (float *)luaL_checkudata(L, 2, "ANN_CONNECTION");
	int delta_n = lua_rawlen(L, 2) / sizeof(float);
	if (base_n != delta_n)
		return luaL_error(L, "accumlate size mismatch");
	int i;
	for (i=0;i<base_n;i++) {
		base[i] += delta[i];
	}
	return 0;
}

static int
lconnection_update(lua_State *L) {
	float *base = (float *)luaL_checkudata(L, 1, "ANN_CONNECTION");
	int base_n = lua_rawlen(L, 1) / sizeof(float);
	float *delta = (float *)luaL_checkudata(L, 2, "ANN_CONNECTION");
	int delta_n = lua_rawlen(L, 2) / sizeof(float);
	if (base_n != delta_n)
		return luaL_error(L, "update size mismatch");
	float eta = luaL_checknumber(L, 3);
	int i;
	for (i=0;i<base_n;i++) {
		base[i] = base[i] - eta * delta[i];
	}
	return 0;
}

static inline float *
weight(float *base, struct connection *c, int output_idx) {
	return base + c->input_n * output_idx + c->output_n;
}

static inline float *
bias(float *base, struct connection *c) {
	(void)c;
	return base;
}

static int
lconnection_dump(lua_State *L) {
	float *f = (float *)luaL_checkudata(L, 1, "ANN_CONNECTION");
	lua_getiuservalue(L, 1, 1);
	struct connection *c = (struct connection *)lua_touserdata(L, -1);
	float *b = bias(f, c);
	int i,j;
	for (i=0;i<c->output_n;i++) {
		float *w = weight(f, c, i);
		printf("[%d] BIAS %g ", i, b[i]);
		for (j=0;j<c->input_n;j++) {
			if (w[j] != 0)
				printf("%d:%g ", j, w[j]);
		}
		printf("\n");
	}
	return 0;
}

static int
lconnection_import(lua_State *L) {
	float *f = (float *)luaL_checkudata(L, 1, "ANN_CONNECTION");
	lua_getiuservalue(L, 1, 1);
	struct connection *c = (struct connection *)lua_touserdata(L, -1);
	luaL_checktype(L, 2, LUA_TTABLE);	// bias
	luaL_checktype(L, 3, LUA_TTABLE);	// weight
	int size_bias = lua_rawlen(L, 2);
	int size_weight = lua_rawlen(L, 3);
	if (size_bias != size_weight && size_bias != c->output_n)
		return luaL_error(L, "Invalid size");
	float *b = bias(f, c);
	int i,j;
	for (i=0;i<c->output_n;i++) {
		if (lua_rawgeti(L, 2, i+1) != LUA_TNUMBER)
			return luaL_error(L, "Invalid bias[%d]", i+1);
		b[i] = lua_tonumber(L, -1);
		lua_pop(L, 1);
	}
	for (i=0;i<c->output_n;i++) {
		if (lua_rawgeti(L, 3, i+1) != LUA_TTABLE)
			return luaL_error(L, "Invalid weight[%d]", i+1);
		int n = lua_rawlen(L, -1);
		if (n != c->input_n)
			return luaL_error(L, "Invalid weight_size[%d]", i+1);
		float *w = weight(f, c, i);
		for (j=0;j<n;j++) {
			if (lua_rawgeti(L, -1, j+1) != LUA_TNUMBER)
				return luaL_error(L, "Invalid weight[%d][%d]", i+1, j+1);
			w[j] = lua_tonumber(L, -1);
			lua_pop(L, 1);
		}
		lua_pop(L, 1);
	}
	return 0;
}

static int
lconnection(lua_State *L) {
	struct connection * c = (struct connection *)lua_newuserdatauv(L, sizeof(*c), 0);
	c->input_n = luaL_checkinteger(L, 1);
	c->output_n = luaL_checkinteger(L, 2);
	size_t sz = (c->input_n * c->output_n + c->output_n) * sizeof(float);
	float * data = (float *)lua_newuserdatauv(L, sz, 1);
	lua_pushvalue(L, -2);
	lua_setiuservalue(L, -2, 1);
	memset(data, 0, sz);
	if (luaL_newmetatable(L, "ANN_CONNECTION")) {
		lua_pushvalue(L, -1);
		lua_setfield(L, -2, "__index");
		luaL_Reg l[] = {
			{ "randn", lconnection_randn },
			{ "accumulate", lconnection_accumulate },
			{ "update", lconnection_update },
			{ "dump", lconnection_dump },
			{ "import", lconnection_import },
			{ NULL, NULL },
		};
		luaL_setfuncs(L, l, 0);
	}
	lua_setmetatable(L, -2);
	return 1;
}

static inline float
sigmoid(float z) {
	return 1.0f / (1.0f + expf(-z));
}

static inline float
sigmoid_prime(float s) {
	return s * (1-s);
}

static int
lfeedforward(lua_State *L) {
	struct layer input = checklayer(L, 1);
	struct layer output = checklayer(L, 2);
	float *f = (float *)luaL_checkudata(L, 3, "ANN_CONNECTION");
	lua_getiuservalue(L, 3, 1);
	struct connection *c = (struct connection *)lua_touserdata(L, -1);
	int i,j;
	for (i=0;i<output.n;i++) {
		float s = 0;
		float *w = weight(f, c, i);
		float *b = bias(f, c);
		for (j=0;j<input.n;j++) {
			s += input.data[j] * w[j];
		}
		output.data[i] = sigmoid(s + b[i]);
	}
	return 0;
}

// https://builtin.com/machine-learning/backpropagation-neural-network

static struct connection *
get_connection(lua_State *L, int idx) {
	lua_getiuservalue(L, idx, 1);
	struct connection *c = (struct connection *)lua_touserdata(L, -1);
	lua_pop(L, 1);
	return c;
}

// [Input]  --(nabla)-->  [result/Expect]
//
// delta := (result - expect) * sigmoid'(result)
// nabla_b := delta
// nabla_w := dot(delta, Input)

static int
lbackprop_last(lua_State *L) {
	struct layer input = checklayer(L, 1);
	struct layer result = checklayer(L, 2);
	struct layer expect = checklayer(L, 3);
	float *delta = (float *)luaL_checkudata(L, 4, "ANN_CONNECTION");
	struct connection *c = get_connection(L, 4);
	if (c->input_n != input.n || c->output_n != result.n) {
		return luaL_error(L, "Invalid output delta");
	}
	float *nabla_b = bias(delta, c);
	int i, j;
	for (i=0;i<c->output_n;i++) {
		float *nabla_w = weight(delta, c, i);
		// cost derivative
		float d = (result.data[i] - expect.data[i]) * sigmoid_prime(result.data[i]);
		nabla_b[i] = d;
		for (j=0;j<input.n;j++) {
			nabla_w[j] = d * input.data[j];
		}
	}
	return 0;
}

// [Input] --(nabla)--> [Z] --(delta_last/conn_output)-->
//
// delta := dot(conn_output_weight, delta_last) * sigmoid'(Z)
// nabla_b := delta
// nabla_w := dot(delta, Input)

static int
lbackprop(lua_State *L) {
	struct layer input = checklayer(L, 1);
	struct layer z = checklayer(L, 2);

	float *delta = (float *)luaL_checkudata(L, 3, "ANN_CONNECTION");
	struct connection *input_c = get_connection(L, 3);

	float *delta_last = (float *)luaL_checkudata(L, 4, "ANN_CONNECTION");
	struct connection *output_c = get_connection(L, 4);

	float *conn_output = (float *)luaL_checkudata(L, 5, "ANN_CONNECTION");

	if (input_c->output_n != output_c->input_n || input_c->output_n != z.n || input_c->input_n != input.n) {
		return luaL_error(L, "input/output mismatch");
	}
	float *nabla_b = bias(delta, input_c);
	int i, j;
	for (i=0;i<z.n;i++) {
		float *nabla_w = weight(delta, input_c, i);
		float d = 0;
		float *delta_last_b = bias(delta_last, output_c);
		for (j=0;j<output_c->output_n;j++) {
			d += delta_last_b[j] * weight(conn_output, output_c, j)[i];
		}
		d *= sigmoid_prime(z.data[i]);
		nabla_b[i] = d;
		for (j=0;j<input.n;j++) {
			nabla_w[j] = d * input.data[j];
		}
	}
	return 0;
}

LUAMOD_API int
luaopen_ann(lua_State *L) {
	luaL_checkversion(L);
	luaL_Reg l[] = {
		{ "layer" , llayer },
		{ "connection", lconnection },
		{ "feedforward", lfeedforward },
		{ "backprop", lbackprop },
		{ "backprop_last", lbackprop_last },
		{ NULL, NULL },
	};
	luaL_newlib(L, l);
	return 1;
}

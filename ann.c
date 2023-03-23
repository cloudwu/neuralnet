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

static int
lsignal_relu(lua_State *L) {
	struct signal * s = check_signal(L, 1);
	int i;
	for (i=0;i<s->n;i++) {
		if (s->data[i] < 0)
			s->data[i] = 0;
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

static int
lsignal_image(lua_State *L) {
	struct signal * s = check_signal(L, 1);
	luaL_Buffer b;
	luaL_buffinit(L, &b);
	int i;
	for (i=0;i<s->n;i++) {
		float v = s->data[i];
		int c;
		if (v <= 0)
			c = 0;
		else if (v >= 1.0f)
			c = 255;
		else
			c = v * 255;
		luaL_addchar(&b, c);
	}
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
			{ "image", lsignal_image },
			{ "init", lsignal_init },
			{ "randn", lsignal_randn },
			{ "max", lsignal_max },
			{ "size", lsignal_size },
			{ "accumulate", lsignal_accumulate },
			{ "sigmoid", lsignal_sigmoid },
			{ "relu", lsignal_relu },
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
lbackprop_sigmoid(lua_State *L) {
	struct signal * s = check_signal(L, 1);
	struct signal * input = check_signal(L, 2);
	if (s->n != input->n)
		return luaL_error(L, "Invalid signal size");
	int i;
	for (i=0;i<s->n;i++) {
		input->data[i] *= sigmoid_prime(s->data[i]);
	}
	return 0;
}

static int
lbackprop_relu(lua_State *L) {
	struct signal * s = check_signal(L, 1);
	struct signal * input = check_signal(L, 2);
	if (s->n != input->n)
		return luaL_error(L, "Invalid signal size");
	int i;
	for (i=0;i<s->n;i++) {
		if (s->data[i] <= 0)
			input->data[i] = 0;
	}
	return 0;
}

static void
softmax(struct signal *a, struct signal *output) {
	int i;
	float m = a->data[0];
	for (i=1;i<a->n;i++) {
		if (a->data[i] > m)
			m = a->data[i];
	}
	float sum = 0;
	for (i=0;i<a->n;i++) {
		float exp_a = expf(a->data[i] - m);
		output->data[i] = exp_a;
		sum += exp_a;
	}
	float inv_sum = 1.0f / sum;
	for (i=0;i<a->n;i++) {
		output->data[i] *= inv_sum;
	}
}


static int
lsignal_softmax(lua_State *L) {
	struct signal * a = check_signal(L, 1);
	struct signal * b = check_signal(L, 2);
	struct signal * output = check_signal(L, 3);
	if (a->n != b->n || a->n != output->n)
		return luaL_error(L, "Invalid signal size");
	softmax(a, output);
	int i;
	for (i=0;i<a->n;i++) {
		output->data[i] -= b->data[i];
	}
	return 0;
}

// filter for convolution with stride 1.
struct filter {
	int size;	// (size * size) filter
	int pooling;
	int n;
	int src_w;
	int src_h;
	float f[1];	// bias[n] + weight[size * size * n]
};

static inline size_t
filter_size(int size, int n) {
	int nfloat = (size * size + 1) * n;
	return sizeof(struct filter) + (nfloat - 1) * sizeof(float);
}

static inline float *
filter_weight(struct filter *f, int n) {
	return f->f + f->n + f->size * f->size * n;
}

static inline float
filter_bias(struct filter *f, int n) {
	return f->f[n];
}

static struct filter *
check_filter(lua_State *L, int index) {
	return luaL_checkudata(L, index, "ANN_FILTER");
}

static int
lfilter_randn(lua_State *L) {
	struct filter *f = check_filter(L, 1);
	float deviation = luaL_optnumber(L, 2, 1.0f);
	int n = f->n * (1 + f->size * f->size);
	randn(f->f, n, deviation);
	lua_settop(L, 1);
	return 1;
}

static int
lfilter_zero(lua_State *L) {
	struct filter *f = check_filter(L, 1);
	int n = f->n * (1 + f->size * f->size);
	memset(f->f, 0, n * sizeof(float));
	lua_settop(L, 1);
	return 1;
}

static int
lfilter_dump(lua_State *L) {
	struct filter * f = check_filter(L, 1);
	int i,j,k;
	luaL_Buffer b;
	luaL_buffinit(L, &b);

	for (i=0; i<f->n; i++) {
		addfloat(L, &b, filter_bias(f, i));
		luaL_addchar(&b, '\n');
		float * w = filter_weight(f, i);
		for (j=0;j<f->size;j++) {
			luaL_addlstring(&b, "  [", 3);
			for (k=0;k<f->size;k++) {
				addfloat(L, &b, *w);
				++w;
			}
			luaL_addlstring(&b, "]\n", 2);
		}
	}
	luaL_pushresult(&b);
	return 1;
}

static void
set_arg(lua_State *L, const char *key, int v) {
	lua_pushinteger(L, v);
	lua_setfield(L, -2, key);
}

static inline void
filter_output_size(struct filter *f, int *w, int *h) {
	*w = f->src_w - f->size + 1;
	*h = f->src_h - f->size + 1;
}

static int
lfilter_args(lua_State *L) {
	struct filter * f = check_filter(L, 1);
	lua_newtable(L);
	set_arg(L, "size", f->size);
	set_arg(L, "n", f->n);
	set_arg(L, "w", f->src_w);
	set_arg(L, "h", f->src_h);
	set_arg(L, "pooling", f->pooling);
	int dw, dh;
	filter_output_size(f, &dw, &dh);
	set_arg(L, "cw", dw);
	set_arg(L, "ch", dh);
	set_arg(L, "conv_size", dw * dh * f->n);
	dw /= f->pooling;
	dh /= f->pooling;
	set_arg(L, "pw", dw);
	set_arg(L, "ph", dh);
	set_arg(L, "output_size", dw * dh * f->n);

	return 1;
}

static inline float
conv_dot(const float *src, int stride, const float *f, int fsize) {
	int i,j;
	float s = 0;
	const float *line = src;
	for (i=0;i<fsize;i++) {
		for (j=0;j<fsize;j++) {
			s += line[j] * (*f);
			++f;
		}
		line += stride;
	}
	return s;
}

static void
conv2dpool(const float *src, int w, int h, float *dst, int fsize, const float *f, float bias) {
	int i,j;
	const float * line = src;
	int y = h - fsize + 1;
	int x = w - fsize + 1;
	for (i=0;i<y;i++) {
		const float * src = line;
		for (j=0;j<x;j++) {
			float v = conv_dot(src, w, f, fsize);
			*dst = v + bias;
			++dst;
			++src;
		}
		line += w;
	}
}

static int
lfilter_convolution(lua_State *L) {
	struct filter *f = check_filter(L, 1);
	struct signal *input = check_signal(L, 2);
	struct signal *output = check_signal(L, 3);

	int input_size = f->src_w * f->src_h;
	int dw,dh;
	filter_output_size(f, &dw, &dh);
	int output_size = dw * dh;
	if (input_size != input->n)
		return luaL_error(L, "Invalid input signal size %d * %d != %d", f->src_w, f->src_h, input->n);
	if (output_size * f->n != output->n)
		return luaL_error(L, "Invalid output signal size %d * %d * %d != %d", dw, dh, f->n, output->n);

	int i;
	float *oimg = output->data;
	for (i=0;i<f->n;i++) {
		conv2dpool(input->data, f->src_w, f->src_h, oimg, f->size, filter_weight(f, i), filter_bias(f, i));
		oimg += output_size;
	}
	return 0;
}

static inline float
pooling_max(const float *src, int x, int y, int pooling, int stride) {
	int i,j;
	src += y * pooling * stride + x * pooling;
	float m = *src;
	for (i=0;i<pooling;i++) {
		for (j=0;j<pooling;j++) {
			float v = src[j];
			if (v > m)
				m = v;
		}
		src += stride;
	}
	return m;
}

static int
lfilter_maxpooling(lua_State *L) {
	struct filter *f = check_filter(L, 1);
	struct signal *input = check_signal(L, 2);
	struct signal *output = check_signal(L, 3);

	int dw,dh;
	filter_output_size(f, &dw, &dh);
	int input_size = dw * dh;
	if (input_size * f->n != input->n)
		return luaL_error(L, "Invalid input signal size %d * %d * %d != %d", dw, dh, f->n, input->n);
	int pw = dw / f->pooling;
	int ph = dh / f->pooling;
	int output_size = pw * ph;
	if (output_size * f->n != output->n)
		return luaL_error(L, "Invalid output signal size %d * %d * %d != %d", pw, ph, f->n, output->n);

	int i,j,k;
	float *ptr = output->data;
	const float * src = input->data;
	int pooling_size = f->pooling;
	for (i=0;i<f->n;i++) {
		for (j=0;j<ph;j++) {
			for (k=0;k<pw;k++) {
				*ptr = pooling_max(src, k, j, pooling_size, dw);
				++ptr;
			}
		}
		src += input_size;
	}
	return 0;
}

static inline void
fill_max(float * conv, float delta, int stride, int pooling) {
	int i,j;
	float * m = conv;
	float maxv = *m;
	for (i=0;i<pooling;i++) {
		for (j=0;j<pooling;j++) {
			float v = conv[j];
			if (v > maxv) {
				maxv = v;
				m = &conv[j];
			}
			conv[j] = 0;
		}
		conv += stride;
	}
	*m = delta;
}

static void
pooling_max_backprop(const float *delta_img, float *conv_img, int w, int h, int pooling) {
	int i,j;
	int y = h - pooling + 1;
	int x = w - pooling + 1;
	int stride = w * pooling;
	for (i=0;i<y;i+=pooling) {
		for (j=0;j<x;j+=pooling) {
			fill_max(conv_img + j , *delta_img, w, pooling);
			delta_img ++;
		}
		for (;j<w;j++) {
			conv_img[j] = 0;
		}
		conv_img += stride;
	}
	memset(conv_img, 0, (h-i) * w * sizeof(float));
}

static inline float
calc_filter_weight(const float * a, int stride, const float *b, int w, int h) {
	float s = 0;
	int i,j;
	for (i=0;i<h;i++) {
		for (j=0;j<w;j++) {
			s += a[j] * (*b);
			++b;
		}
		a += stride;
	}
	return s;
}

// https://microsoft.github.io/ai-edu/%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B/A2-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86/%E7%AC%AC8%E6%AD%A5%20-%20%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/17.3-%E5%8D%B7%E7%A7%AF%E7%9A%84%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD%E5%8E%9F%E7%90%86.html

static int
lbackprop_conv_bias(lua_State *L) {
	struct filter *f = check_filter(L, 1);
	struct signal *delta = check_signal(L, 2);

	int size = delta->n / f->n;
	const float * ptr = delta->data;
	int i,j;
	for (i=0;i<f->n;i++) {
		float s = 0;
		for (j=0;j<size;j++) {
			s += *ptr;
			++ptr;
		}
		f->f[i] = s;
	}

	return 0;
}

static int
lbackprop_maxpooling(lua_State *L) {
	struct filter *f = check_filter(L, 1);
	struct signal *conv =  check_signal(L, 2);
	struct signal *delta = check_signal(L, 3);

	int dw,dh;
	filter_output_size(f, &dw, &dh);
	int conv_size = dw * dh;

	int pw = dw / f->pooling;
	int ph = dh / f->pooling;
	int output_size = pw * ph;

	if (conv_size * f->n != conv->n)
		return luaL_error(L, "Invalid input convolution size %d * %d != %d", dw, dh, f->n, conv->n);

	if (output_size * f->n != delta->n)
		return luaL_error(L, "Invalid output signal size %d * %d * %d != %d", pw, ph, f->n, delta->n);

	int i;
	const float * delta_img = delta->data;
	float * conv_img = conv->data;
	for (i=0;i<f->n;i++) {
		pooling_max_backprop(delta_img, conv_img, dw, dh, f->pooling);
		delta_img += output_size;
		conv_img += conv_size;
	}

	return 0;
}

static int
lbackprop_conv_weight(lua_State *L) {
	struct filter *f = check_filter(L, 1);
	struct signal *input = check_signal(L, 2);
	struct signal *delta =  check_signal(L, 3);

	int input_size = f->src_w * f->src_h;
	int dw,dh;
	filter_output_size(f, &dw, &dh);
	int delta_size = dw * dh;

	if (input_size != input->n)
		return luaL_error(L, "Invalid input signal size %d * %d != %d", f->src_w, f->src_h, input->n);

	if (delta_size * f->n != delta->n)
		return luaL_error(L, "Invalid input delta size %d * %d != %d", dw, dh, f->n, delta->n);

	const float * input_img = input->data;
	float * delta_img = delta->data;
	int i,j,k;
	for (i=0;i<f->n;i++) {
		const float * line = input_img;
		float * w = filter_weight(f, i);
		for (j=0;j<f->size;j++) {
			for (k=0;k<f->size;k++) {
				*w = calc_filter_weight(line + k, f->src_w, delta_img, dw, dh);
				++w;
			}
			line += f->src_w;
		}
		delta_img += delta_size;
	}

	return 0;
}

static int
lfilter_clone(lua_State *L) {
	struct filter *f = check_filter(L, 1);
	size_t sz = lua_rawlen(L, 1);
	void * c = lua_newuserdatauv(L, sz, 0);
	memcpy(c, f, sz);
	lua_getmetatable(L, 1);
	lua_setmetatable(L, -2);

	return 1;
}

static int
lfilter_accumulate(lua_State *L) {
	struct filter * f = check_filter(L, 1);
	struct filter * delta = check_filter(L, 2);
	if (f->size != delta->size || f->n != delta->n)
		return luaL_error(L, "filter size (%d , %d) != (%d , %d)", f->size, f->n, delta->size, delta->n);
	int i;
	int nfloat = (f->size * f->size + 1) * f->n;
	if (lua_type(L, 3) == LUA_TNUMBER) {
		float eta = lua_tonumber(L, 3);
		for (i=0;i<nfloat;i++) {
			f->f[i] += delta->f[i] * eta;
		}
	} else {
		for (i=0;i<nfloat;i++) {
			f->f[i] += delta->f[i];
		}
	}
	lua_settop(L, 1);
	return 1;
}

static int
lfilter_export(lua_State *L) {
	struct filter * f = check_filter(L, 1);
	lua_createtable(L, f->n, 0);
	int i,j;
	int size = f->size * f->size;
	for (i=0;i<f->n;i++) {
		lua_createtable(L, size, 1);
		const float * w = filter_weight(f, i);
		for (j=0;j<size;j++) {
			lua_pushnumber(L, w[j]);
			lua_rawseti(L, -2, j+1);
		}
		lua_pushnumber(L, filter_bias(f, i));
		lua_setfield(L, -2, "bias");
		lua_rawseti(L, -2, i+1);
	}
	return 1;
}

static int
lfilter_import(lua_State *L) {
	struct filter * f = check_filter(L, 1);
	luaL_checktype(L, 2, LUA_TTABLE);
	int i,j;
	int size = f->size * f->size;
	for (i=0;i<f->n;i++) {
		if (lua_rawgeti(L, 2, i+1) != LUA_TTABLE)
			return luaL_error(L, "[%d] is not a table (%s)", i+1, lua_typename(L, lua_type(L, -1)));
		if (lua_getfield(L, -1, "bias") != LUA_TNUMBER) {
			return luaL_error(L, "[%d] missing bias", i+1);
		}
		f->f[i] = lua_tonumber(L, -1);
		lua_pop(L, 1);
		float *w = filter_weight(f, i);
		for (j=0;j<size;j++) {
			if (lua_rawgeti(L, -1, j+1) != LUA_TNUMBER) {
				return luaL_error(L, "[%d, %d] is not a number (%s)", i+1, j+1, lua_typename(L, lua_type(L, -1)));
			}
			w[j] = lua_tonumber(L, -1);
			lua_pop(L, 1);
		}
		lua_pop(L, 1);
	}
	return 0;
}

static int
lconvpool_filter(lua_State *L) {
	int size = luaL_checkinteger(L, 1);
	int src_w = luaL_checkinteger(L, 2);
	int src_h = luaL_checkinteger(L, 3);
	int n = luaL_checkinteger(L, 4);
	int pooling = luaL_optinteger(L, 5, 2);
	size_t sz = filter_size(size, n);
	struct filter * f = (struct filter *)lua_newuserdatauv(L, sz, 0);
	memset(f, 0, sz);
	f->size = size;
	f->n = n;
	f->src_w = src_w;
	f->src_h = src_h;
	f->pooling = pooling;

	if (luaL_newmetatable(L, "ANN_FILTER")) {
		lua_pushvalue(L, -1);
		lua_setfield(L, -2, "__index");
		luaL_Reg l[] = {
			{ "clone", lfilter_clone },
			{ "accumulate", lfilter_accumulate },
			{ "randn", lfilter_randn },
			{ "zero", lfilter_zero },
			{ "__tostring", lfilter_dump },
			{ "args", lfilter_args },
			{ "convolution", lfilter_convolution },
			{ "maxpooling", lfilter_maxpooling },
			{ "export", lfilter_export },
			{ "import", lfilter_import },
			{ "backprop_maxpooling", lbackprop_maxpooling },
			{ "backprop_conv_bias", lbackprop_conv_bias},
			{ "backprop_conv_weight", lbackprop_conv_weight},
			{ NULL, NULL },
		};
		luaL_setfuncs(L, l, 0);
	}
	lua_setmetatable(L, -2);

	return 1;
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
		{ "softmax_error", lsignal_softmax },
		{ "backprop_sigmoid", lbackprop_sigmoid },
		{ "backprop_relu", lbackprop_relu },
		{ "convpool_filter", lconvpool_filter },
		{ NULL, NULL },
	};
	luaL_newlib(L, l);
	return 1;
}

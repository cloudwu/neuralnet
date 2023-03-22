local mnist = require "mnist"
local ann = require "ann"

local labels = mnist.labels "data/train-labels.idx1-ubyte"
local images = mnist.images "data/train-images.idx3-ubyte"

local network = {}	; network.__index = network

function network.new(args)
	local n = {
		input = ann.signal(args.input),
		hidden = ann.signal(args.hidden),
		output = ann.signal(args.output),
		weight_ih = ann.weight(args.input, args.hidden):randn(),
		weight_ho = ann.weight(args.hidden, args.output):randn(),
		bias_hidden = ann.signal(args.hidden):randn(),
		bias_output = ann.signal(args.output):randn(),
	}

	return setmetatable(n, network)
end

function network:feedforward(image)
	self.input:init(image)
	ann.prop(self.input, self.hidden, self.weight_ih)
	self.hidden:accumulate(self.bias_hidden):sigmoid()
	ann.prop(self.hidden, self.output, self.weight_ho)
	return self.output:accumulate(self.bias_output)
end

local function shffule_training_data(t)
	local n = #t
	for i = 1, n - 1 do
		local r = math.random(i, n)
		t[i] , t[r] = t[r], t[i]
	end
end

function network:train(training_data, batch_size, eta)
	shffule_training_data(training_data)

	local eta_ = - eta / batch_size
	local dw_ih = ann.weight(self.weight_ih:size())
	local dw_ih_s = ann.weight(self.weight_ih:size())
	local dw_ho = ann.weight(self.weight_ho:size())
	local dw_ho_s = ann.weight(self.weight_ho:size())
	local db_output_s = ann.signal(self.output:size())
	local db_hidden = ann.signal(self.hidden:size())
	local db_hidden_s = ann.signal(self.hidden:size())

	local db_output

	local function backprop(expect)
		-- calc error
		ann.softmax_error(self.output, expect, db_output)
		-- backprop from output to hidden
		ann.backprop_weight(self.hidden, db_output, dw_ho)
		ann.backprop_bias(db_hidden, db_output, self.weight_ho)
		ann.backprop_sigmoid(self.hidden, db_hidden)
		-- backprop from hidden to input
		ann.backprop_weight(self.input, db_hidden, dw_ih)
	end

	for i = 1, #training_data, batch_size do
		self:feedforward(training_data[i].image)
		db_output = db_output_s
		backprop(training_data[i].expect)
		db_output = self.output
		db_hidden_s, db_hidden = db_hidden, db_hidden_s
		dw_ih_s, dw_ih = dw_ih, dw_ih_s
		dw_ho_s, dw_ho = dw_ho, dw_ho_s

		for j = 1, batch_size-1 do
			local image = training_data[i+j]
			if image then
				self:feedforward(image.image)
				backprop(training_data[i+j].expect)
				dw_ih_s:accumulate(dw_ih)
				dw_ho_s:accumulate(dw_ho)
				db_output_s:accumulate(db_output)
				db_hidden_s:accumulate(db_hidden)
			else
				eta_ = - eta / j
				break
			end
		end

		self.weight_ih:accumulate(dw_ih_s, eta_)
		self.weight_ho:accumulate(dw_ho_s, eta_)
		self.bias_hidden:accumulate(db_hidden_s, eta_)
		self.bias_output:accumulate(db_output_s, eta_)
	end
end

local function gen_training_data()
	local result = {}
	for i = 0, 9 do
		result[i] = ann.signal(10):init(i)
	end
	local training = {}
	for i = 1, #images do
		training[i] = {
			image = images[i],
			expect = result[labels[i] ],
			value = labels[i],
		}
	end
	return training
end

local n = network.new {
	input = images.row * images.col,
	hidden = 30,
	output = 10,
}

local data = gen_training_data()

local labels = mnist.labels "data/t10k-labels.idx1-ubyte"
local images = mnist.images "data/t10k-images.idx3-ubyte"

local function test()
	local s = 0
	for idx = 1, #labels do
		local r, p = n:feedforward(images[idx]):max()
		local label = labels[idx]
		if r~=label then
			s = s + 1
		end
	end
	return (s / #labels * 100) .."%"
end

for i = 1, 30 do
	n:train(data,20,3.0)
	print("Epoch", i, test())
end

local mnist = require "mnist"
local ann = require "ann"
local images = mnist.images "data/train-images.idx3-ubyte"

local x, y = images.col, images.row

local image = ann.signal(x * y):init(images[1])

local filter = ann.convpool_filter(
	3,	-- size
	x,y,
	2,	-- filter number
	2)	-- pooling 2x2

filter:import {
	{ bias = 0,
		0, -1, 0,
	    0, 2, 0,
		0, -1, 0 } ,
	{ bias = 0.1,
		0, -0.5, 0,
	    0, 1, 0,
		0, -0.5, 0 }
}

local args = filter:args()
local conv = ann.signal(args.conv_size)
local result = ann.signal(args.output_size)
local expect = ann.signal(args.output_size)

filter:convolution(image, conv)
filter:maxpooling(conv, expect)

--[[
local f = assert(io.open("conv.pgm", "wb"))
f:write(mnist.pgm(expect:image(), args.pw, args.ph * 2))
f:close()
]]

filter:randn()

local delta = filter:clone()
local scale = - 0.5 / args.output_size

for i = 1, 20000 do
	filter:convolution(image, conv)
	filter:maxpooling(conv, result)
	result:accumulate(expect, -1)		-- error

	delta:backprop_conv_bias(result)
	delta:backprop_maxpooling(conv, result)
	delta:backprop_conv_weight(image, conv)

	filter:accumulate(delta, scale)
end

print(filter)

filter:convolution(image, conv)
filter:maxpooling(conv, expect)
--[[
local f = assert(io.open("conv2.pgm", "wb"))
f:write(mnist.pgm(expect:image(), args.pw, args.ph * 2))
f:close()
]]
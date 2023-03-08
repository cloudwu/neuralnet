local mnist = require "mnist"

local labels = mnist.labels "data/train-labels.idx1-ubyte"
print(#labels)

for i = 1, 10 do
	print(labels[i])
end

local images = mnist.images "data/train-images.idx3-ubyte"
local image = images[1]
local size = images.row * images.col

local f = assert(io.open("image1.pgm", "wb"))
f:write(mnist.pgm(image, images.row, images.col))
f:close()


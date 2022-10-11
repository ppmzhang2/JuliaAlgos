using Flux

include("utils.jl")

"""
    buildmodel(inwidth::Int64, inheight::Int64, inchannel::Int64,
        nclass::Int64, poolsize::Int64, filtersize::Int64, outchannel::Int64)

build a DNN model
"""
function getmodel(inwidth::Int64, inheight::Int64, inchannel::Int64,
    nclass::Int64, poolsize::Int64, filtersize::Int64, outchannel::Int64)
    outdim = let r = poolsize^3
        Int.(floor.([inwidth / r, inheight / r, outchannel]))
    end

    Chain(
        # First convolution, operating upon a 28x28 image
        Conv((filtersize, filtersize), inchannel => 16, pad=(1, 1), relu),
        MaxPool((poolsize, poolsize)),

        # Second convolution, operating upon a 14x14 image
        Conv((filtersize, filtersize), 16 => outchannel, pad=(1, 1), relu),
        MaxPool((poolsize, poolsize)),

        # Third convolution, operating upon a 7x7 image
        Conv((filtersize, filtersize), outchannel => outchannel, pad=(1, 1),
            relu),
        MaxPool((poolsize, poolsize)),

        # Reshape 3d tensor into a 2d one using `Flux.flatten`, at this point
        # it should be (3, 3, 32, N)
        Flux.flatten,
        Dense(prod(outdim), nclass),
        softmax,
    )
end

getpredictor(model::Chain) = x -> (x |> model |> onecold) .- 1

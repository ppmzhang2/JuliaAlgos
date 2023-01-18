include("../mldata/MLData.jl")

using .MLData

"""
    getdata(; test::Bool = true)

get feature and target for training or testing
"""
function getdata(; test::Bool=true)
    x, y = load_mnist(test=test)
    xsize = x |> size
    return reshape(x, xsize[1], xsize[2], 1, :), y
end

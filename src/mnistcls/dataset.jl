using Flux, MLDatasets, StatsBase


include("utils.jl")

"""
    getloader(batch::Int64)

build a mini-batch loader with `Flux.Data.DataLoader`
"""
function getloader(batch::Int64)
    dataloader(data::MNIST) =
        let ft = data.features
            targets = (data.targets .+ 1) |> onehot
            Flux.Data.DataLoader(
                (reshape(ft, size(ft)[1], size(ft)[2], 1, :), targets),
                batchsize=batch, shuffle=true)
        end

    MNIST(:train) |> dataloader
end

"""
    getdata(test::Bool = true)

get feature and target for training or testing
"""
function getdata(test::Bool=true)
    helper(data::MNIST) =
        let ft = data.features
            targets = data.targets
            reshape(ft, size(ft)[1], size(ft)[2], 1, :), targets
        end

    if test
        MNIST(:test) |> helper
    else
        MNIST(:train) |> helper
    end
end

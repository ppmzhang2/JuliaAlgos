module MnistClassifier
export Trainer

using BSON: @save
using BSON, Flux


include("dataset.jl")
include("measures.jl")
include("model.jl")

const DATA_PATH = normpath(joinpath(@__DIR__, "..", "..", "data"))


"""
    callable training arguments
"""
Base.@kwdef struct Trainer
    inwidth::Int64 = 28
    inheight::Int64 = 28
    inchannel::Int64 = 1
    nclass::Int64 = 10
    poolsize::Int64 = 2
    filtersize::Int64 = 3
    outchannel::Int64 = 32
    η::Float64 = 1e-3
    nepoch::Int64 = 5
    batchsize::Int64 = 2^5
    modelpath::String = joinpath(DATA_PATH, "mnist.bson")
end

"""
start training
"""
function (args::Trainer)()
    model = getmodel(args.inwidth, args.inheight, args.inchannel, args.nclass,
        args.poolsize, args.filtersize, args.outchannel)
    loader = getloader(args.batchsize)
    xtest, ytest = getdata()
    optimizer = Adam(args.η, (0.9, 0.99), 1.0e-8)

    accbest = 0.0
    acc = 0.0
    for epoch in 1:args.nepoch
        @info("training loop $epoch / $(args.nepoch) starting ...")
        for (x, y) in loader
            Flux.train!(lossfunc(model), Flux.params(model), ((x, y),),
                optimizer)
        end

        @info("  -- evaluating ...")
        acc = accuracy(getpredictor(model)(xtest), ytest)
        @info("  -- accuracy=$acc")

        if acc >= accbest
            @info("  -- new best record")
            accbest = acc
            @save args.modelpath model
        end

        if accbest >= 0.99
            @info("  -- early exit")
            break
        end
    end
end

end # module MnistClassifier

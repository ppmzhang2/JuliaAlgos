using Flux

"""
    lossfunc(model::Chain)

build loss function which accepts feature and target
"""
lossfunc(model::Chain) =
    (x::AbstractArray, y::AbstractArray) ->
        Flux.Losses.logitcrossentropy(model(x), y)

"""
    accuracy(ŷ::Vector{<:Integer}, y::Vector{<:Integer}) =

build accuracy function which accepts feature and target
"""
accuracy(ŷ::Vector{<:Integer}, y::Vector{<:Integer}) =
    (ŷ .== y) |> mean

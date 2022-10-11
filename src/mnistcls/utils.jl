using StatsBase

"""
    onehot(data::Vector{<:Integer}; nclass::Number=10)

convert index numbers to onehot arrays
"""
function onehot(data::Vector{<:Integer}; nclass::Number=10)
    1.0f0 * indicatormat(data, nclass)
end


"""
    onecold(array::Matrix{<:AbstractFloat})

convert one-hot matrices into index numbers
"""
function onecold(array::Matrix{<:AbstractFloat})
    map(idx -> idx.I[1], findmax(array, dims=1)[2][1, :])
end

import DataSets, DelimitedFiles, Downloads

const SRCTEST = "https://pjreddie.com/media/files/mnist_test.csv"
const SRCTRAIN = "https://pjreddie.com/media/files/mnist_train.csv"
const DSTTEST = normpath(
    joinpath(@__DIR__, "..", "..", "data", "mnist_test.csv"))
const DSTTRAIN = normpath(
    joinpath(@__DIR__, "..", "..", "data", "mnist_train.csv"))
const TESTNAME = "test"
const TRAINNAME = "train"
const TESTNUM = 10000
const TRAINNUM = 60000
const PROJPATH = normpath(joinpath(@__DIR__, "mnist.toml"))

struct Args
    src::String
    dest::String
    name::String
    num::Int64
end

args_train = Args(SRCTRAIN, DSTTRAIN, TRAINNAME, TRAINNUM)
args_test = Args(SRCTEST, DSTTEST, TESTNAME, TESTNUM)

function _download_mnist(src::String, dst::String)
    if !isfile(dst)
        run(`rm -rf "$dst"`)
        run(`touch "$dst"`)
        # Downloads.download(SRCTRAIN, DSTTRAIN)
        Downloads.download(src, dst)
    end
end

project = DataSets.load_project(PROJPATH)

function _load_mnist(project::DataSets.DataProject, test::Bool)
    args = test ? args_test : args_train
    _download_mnist(args.src, args.dest)
    buf = open(IO, DataSets.dataset(project, args.name))
    raw = DelimitedFiles.readdlm(buf, ',')
    x2d = convert(Matrix{Float32}, raw[:, 2:785])
    y = convert(Vector{Int64}, raw[:, 1])
    # add dimension as `numobs * width * height`
    x3d = reshape(x2d, args.num, 28, 28)
    # permute dimension so that the numobs at the end
    return permutedims(x3d, (3, 2, 1)), y
end

load_mnist(; test::Bool) = _load_mnist(project, test)

struct Point{T<:Real}
    x::T
    y::T
end

Base.:(==)(p1::Point, p2::Point) = (p1.x == p2.x) && (p1.y == p2.y)
Base.:<(p1::Point, p2::Point) = (p1.x < p2.x) && (p1.y < p2.y)
Base.:≤(p1::Point, p2::Point) = (p1.x ≤ p2.x) && (p1.y ≤ p2.y)
Base.:>(p1::Point, p2::Point) = (p1.x > p2.x) && (p1.y > p2.y)
Base.:≥(p1::Point, p2::Point) = (p1.x ≥ p2.x) && (p1.y ≥ p2.y)
Base.:-(p1::Point, p2::Point) = Point(p1.x - p2.x, p1.y - p2.y)

_area(p::Point, origin::Point) =
    if p > origin
        return p.x * p.y
    else
        return -abs(p.x * p.y)
    end

area(p::Point{<:Integer}) = _area(p, Point(0, 0))
area(p::Point{<:AbstractFloat}) = _area(p, Point(0.0, 0.0))

rectarea(p_lower::Point, p_upper::Point) =
    area(p_upper - p_lower)

abstract type RectBox end

struct DiagBox <: RectBox
    lower::Point
    upper::Point
end

struct BoundingBox{T<:Real} <: RectBox
    x::T  # x coordinate of the centre
    y::T  # y coordinate of the centre
    w::T  # width of the box
    h::T  # height of the box
end

lower(db::DiagBox)::Point = db.lower
lower(bb::BoundingBox)::Point =
    let halfw = 0.5 * bb.w
        halfh = 0.5 * bb.h
        Point((bb.x - halfw), (bb.y - halfh))
    end
upper(db::DiagBox)::Point = db.upper
upper(bb::BoundingBox)::Point =
    let halfw = 0.5 * bb.w
        halfh = 0.5 * bb.h
        Point((bb.x + halfw), (bb.y + halfh))
    end
center(db::DiagBox)::Point = Point(
    (db.lower.x + db.upper.x) / 2,
    (db.lower.y + db.upper.y) / 2)
center(bb::BoundingBox)::Point = Point(bb.x, bb.y)

p1(bb::RectBox) = lower(bb)
p3(bb::RectBox) = upper(bb)
p2(bb::RectBox) = Point(upper(bb).x, lower(bb).y)
p4(bb::RectBox) = Point(lower(bb).x, upper(bb).y)

area(b::RectBox) = rectarea(lower(b), upper(b))

interarea(b1::RectBox, b2::RectBox) =
    let plower = Point(
            max(lower(b1).x, lower(b2).x),
            max(lower(b1).y, lower(b2).y))
        pupper = Point(
            min(upper(b1).x, upper(b2).x),
            min(upper(b1).y, upper(b2).y))
        rectarea(plower, pupper)
    end
convexarea(b1::RectBox, b2::RectBox) =
    let plower = Point(
            min(lower(b1).x, lower(b2).x),
            min(lower(b1).y, lower(b2).y))
        pupper = Point(
            max(upper(b1).x, upper(b2).x),
            max(upper(b1).y, upper(b2).y))
        rectarea(plower, pupper)
    end

Base.:∩(b1::DiagBox, b2::DiagBox) = DiagBox(
    Point(max(lower(b1).x, lower(b2).x), max(lower(b1).y, lower(b2).y)),
    Point(min(upper(b1).x, upper(b2).x), min(upper(b1).y, upper(b2).y))
)


function iou(b1::RectBox, b2::RectBox)
    # no negative area
    inter = max(interarea(b1, b2), 0.0)
    b1area = b1 |> area
    b2area = b2 |> area
    iou = inter ./ (b1area .+ b2area .- inter)
    return iou
end

function giou(b1::RectBox, b2::RectBox)
    inter = max(interarea(b1, b2), 0.0)
    convex = convexarea(b1, b2)
    b1area = b1 |> area
    b2area = b2 |> area
    unionarea = b1area .+ b2area .- inter
    inter ./ unionarea - (convex - unionarea) / convex
end

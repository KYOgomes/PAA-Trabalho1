#include "quadtree.hpp"

Quadtree::Quadtree(Rect boundary, int capacity)
    : boundary(boundary), capacity(capacity), divided(false) {}

bool Quadtree::insert(const Point& p) {
    if (!boundary.contains(p)) return false;

    if (points.size() < capacity) {
        points.push_back(p);
        return true;
    } else {
        if (!divided) subdivide();

        if (northeast->insert(p)) return true;
        if (northwest->insert(p)) return true;
        if (southeast->insert(p)) return true;
        if (southwest->insert(p)) return true;
    }
    return false;
}

void Quadtree::subdivide() {
    float x = boundary.x;
    float y = boundary.y;
    float hw = boundary.halfW / 2.0f;
    float hh = boundary.halfH / 2.0f;

    northeast = std::make_unique<Quadtree>(Rect{x+hw, y-hh, hw, hh}, capacity);
    northwest = std::make_unique<Quadtree>(Rect{x-hw, y-hh, hw, hh}, capacity);
    southeast = std::make_unique<Quadtree>(Rect{x+hw, y+hh, hw, hh}, capacity);
    southwest = std::make_unique<Quadtree>(Rect{x-hw, y+hh, hw, hh}, capacity);

    divided = true;
}

void Quadtree::queryRange(const Rect& range, std::vector<Point>& found) const {
    if (!boundary.intersects(range)) return;

    for (const auto& p : points) {
        if (range.contains(p)) {
            found.push_back(p);
        }
    }

    if (divided) {
        northeast->queryRange(range, found);
        northwest->queryRange(range, found);
        southeast->queryRange(range, found);
        southwest->queryRange(range, found);
    }
}

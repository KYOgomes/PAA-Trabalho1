#pragma once
#include "descritor.hpp"
#include <vector>
#include <memory>

struct Point {
    float x, y; // muH, muS
    Record record;
};

struct Rect {
    float x, y;     // centro
    float halfW, halfH; // metade da largura e altura
    bool contains(const Point& p) const {
        return (p.x >= x - halfW && p.x <= x + halfW &&
                p.y >= y - halfH && p.y <= y + halfH);
    }
    bool intersects(const Rect& range) const {
        return !(range.x - range.halfW > x + halfW ||
                 range.x + range.halfW < x - halfW ||
                 range.y - range.halfH > y + halfH ||
                 range.y + range.halfH < y - halfH);
    }
};

class Quadtree {
private:
    Rect boundary;
    int capacity;
    std::vector<Point> points;
    bool divided;

    std::unique_ptr<Quadtree> northeast;
    std::unique_ptr<Quadtree> northwest;
    std::unique_ptr<Quadtree> southeast;
    std::unique_ptr<Quadtree> southwest;

public:
    Quadtree(Rect boundary, int capacity);

    bool insert(const Point& p);
    void subdivide();

    void queryRange(const Rect& range, std::vector<Point>& found) const;
};

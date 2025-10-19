#pragma once

#include <memory>

// ---------------------------------------
// Structs -------------------------------
// ---------------------------------------

typedef struct {
    int x;
    int y;
    int w;
    int h;
} xywh_t;

typedef struct {
    int x1;
    int y1;
    int x2;
    int y2;
} xyxy_t;

typedef struct {
    int x_pad;
    int y_pad;
    float scale;
} letterbox_t;

// ---------------------------------------
// Classes -------------------------------
// ---------------------------------------

class Box
{
public:
    Box(int x1, int y1, int x2, int y2, int class_id, float conf_score);
    ~Box();

    xyxy_t xyxy_;
    xywh_t xywh_;
    int   class_id_;
    float conf_score_;
    
};

class IKalmanFilter; // Forward declaration
class Track
{
public:
    Track(Box b, int track_id, std::unique_ptr<IKalmanFilter> filter);
    ~Track();
    void update(Box& b);
    int getX() const;
    int getY() const;
    
    Box bbox_;
    std::unique_ptr<IKalmanFilter> kf_;
    long track_id_;
    int age_ = 0;

};

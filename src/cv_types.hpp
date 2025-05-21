#pragma once

#include <iostream>

#include <opencv2/opencv.hpp>

typedef struct {
    int x;
    int y;
    int w;
    int h;
} xywh;

typedef struct {
    int x1;
    int y1;
    int x2;
    int y2;
} xyxy;

class Box
{
public:
    Box(int x1, int y1, int x2, int y2, int class_id, float conf_score);
    ~Box();
    void print();

    xyxy xyxy_;
    xywh xywh_;
    
    int   class_id_;
    float conf_score_;
    
};

class Track
{
public:
    Track(Box b);
    ~Track();
    void update();
    void correction(const Box& b);
    int get_xhat() const;
    int get_yhat() const;

    long id_;
    int age_;
    Box bbox_;
    cv::KalmanFilter kf_;

};

typedef struct {
    int x_pad;
    int y_pad;
    float scale;
} letterbox_t;
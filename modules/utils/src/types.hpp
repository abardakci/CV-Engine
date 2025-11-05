#pragma once

namespace forge
{
    typedef struct
    {
        float x;
        float y;
        float w;
        float h;
    } xywh_t;

    typedef struct
    {
        float x1;
        float y1;
        float x2;
        float y2;
    } xyxy_t;

    typedef struct
    {
        int x_pad;
        int y_pad;
        float scale;
    } letterbox_t;

    struct Image
    {
        unsigned char *data;
        int width;
        int height;
        int channels;
    };

    class Box
    {
    public:
        Box(float x1, float y1, float x2, float y2, int class_id, float conf_score)
            : xyxy_{x1, y1, x2, y2},
              xywh_{(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1} {};

        ~Box() = default;

    public:
        xyxy_t xyxy_;
        xywh_t xywh_;
        int class_id_;
        float conf_score_;
    };

}

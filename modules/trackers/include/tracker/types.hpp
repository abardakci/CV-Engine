// ---------------------------------------
// Track Class Implementation
// ---------------------------------------

Track::Track(Box bbox, int track_id, std::unique_ptr<IKalmanFilter> kf)
    : bbox_(bbox), kf_(std::move(kf)), track_id_(track_id), age_(0)
{
    std::vector<float> current_state{bbox_.xywh_.x, bbox_.xywh_.y};
    kf_->init(current_state);
}

void Track::update(Box &true_bbox)
{
    std::vector<float> measurement{true_bbox.xywh_.x, true_bbox.xywh_.y};
    kf_->correct(measurement);
    bbox_ = std::move(true_bbox);
    age_ = 0;
}

class IKalmanFilter; // Forward declaration
class Track
{
public: // constructors and operators
    Track(Box b, int track_id, std::unique_ptr<IKalmanFilter> filter);
    ~Track() = default;
    Track(const Track &) = delete;
    Track &operator=(const Track &) = delete;

    Track(Track &&other) noexcept
        : bbox_(other.bbox_), kf_(std::move(other.kf_)), track_id_(other.track_id_), age_(other.age_) {}

    Track &operator=(Track &&other) noexcept
    {
        if (this != &other)
        {
            this->kf_ = std::move(other.kf_);
            this->track_id_ = other.track_id_;
            this->bbox_ = other.bbox_;
            this->age_ = other.age_;
        }

        return *this;
    }

public: // functions
    void update(Box &b);

public: // variables
    Box bbox_;
    std::unique_ptr<IKalmanFilter> kf_;
    long track_id_;
    int age_ = 0;
};

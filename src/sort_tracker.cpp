#include "sort_tracker.hpp" 

static cv::Mat computeCostMatrix(const std::vector<Track>& tracks, const std::vector<Box>& detects)
{
    int w = detects.size(); // Sütun sayısı (tespitler)
    int h = tracks.size();   // Satır sayısı (izler)

    cv::Mat cost_mat(cv::Size(w, h), CV_32F); // Maliyet matrisi oluştur

    for (int i = 0; i < h; ++i) // Her iz için
    {
        for (int j = 0; j < w; ++j) // Her tespit için
        {
            // İz tahmini ile tespit arasındaki Öklid mesafesini hesapla
            float d = euclidianDistance(detects[j].xywh_.x, detects[j].xywh_.y, tracks[i].get_xhat(), tracks[i].get_yhat());
            cost_mat.at<float>(i, j) = d; // Maliyet matrisine ata
        }
    }

    return cost_mat;
}

// Tracker sınıfının yapıcı metodu
Tracker::Tracker() : highest_id_(0) {} // highest_id_ başlatıldı

// Tracker sınıfının yıkıcı metodu
Tracker::~Tracker() {}

void Tracker::predict() 
{
    for (auto& t : tracks_)
    {
        t.kf_.predict();
    }
}

void Tracker::SORT(std::vector<Box>& detects)
{
// Input: detections (bounding boxes) for current frame
// State: active_tracks ← list of existing Kalman filters

// 1. Predict next position of all active_tracks using Kalman Filter
    predict();

    // 2. Tahmin edilen izler ve mevcut tespitler arasında maliyet matrisini hesapla
    //    Maliyet = Öklid mesafesi (tahmin edilen kutu, tespit kutusu)
    const float threshold = 300; // Eşleşme için maksimum kabul edilebilir mesafe eşiği
    cv::Mat cost_mat = computeCostMatrix(tracks_, detects, threshold);
    
    // Eğer maliyet matrisi boşsa (iz veya tespit yok), erken çıkış yap
    if (cost_mat.empty() || cost_mat.rows == 0 || cost_mat.cols == 0) 
    {
        // Unmatched detections will be created as new tracks
        for (int i = 0; i < detects.size(); ++i)
        {
            tracks_.emplace_back(std::move(detects[i]), ++highest_id_);
        }

        return;
    }

    // 3. Macar Algoritmasını uygulayarak izler ve tespitler arasında optimal eşleşmeleri bul
    std::vector<int> assignments(tracks_.size(), -1); // İz -> Tespit eşleşmelerini tutar (-1 eşleşmeme)
    HungarianAlgorithm(cost_mat, assignments);

    // Eşleşen tespitleri işaretlemek için bir vektör
    std::vector<bool> used_detections(detects.size(), false);

    for (int i = 0; i < assignments.size(); ++i)
    {
        std::cout << assignments[i] << std::endl;
    }

    // 4. Her eşleşen çift (iz_i, tespit_j) için:
    //    a. tespit_j'yi kullanarak iz_i'yi güncelle (Kalman filtresi düzeltmesi)
    //    b. tespit_j'yi kullanıldı olarak işaretle
    for (int i = 0; i < assignments.size(); ++i) // İzler üzerinde döngü
    {
        int j = assignments[i]; // İz 'i' için atanan tespit indeksi

        // ÖNEMLİ DÜZELTME: j'nin geçerli bir indeks olup olmadığını kontrol et
        // ve j'nin 'detects' vektörünün sınırları içinde olduğundan emin ol.
        if (j != -1 && j < detects.size())
        {
            // Eşleşmenin maliyet eşiğinin altında olup olmadığını kontrol et
            if (i < cost_mat.rows && j < cost_mat.cols && cost_mat.at<float>(i, j) <= threshold)
            {
                tracks_[i].age_ = 0; // Eşleşen iz için yaşı sıfırla
                tracks_[i].correction(detects[j]); // İz'i tespit ile düzelt
                tracks_[i].bbox_ = std::move(detects[j]); // Sınırlayıcı kutuyu güncelle
                used_detections[j] = true; // Bu tespiti kullanıldı olarak işaretle
            }
            else
            {
                // Maliyet eşiğin üzerindeyse, eşleşmemiş olarak kabul et
                tracks_[i].age_ += 1; // Eşleşmeyen iz için yaşı artır
            }
        }
        else
        {
            // İz 'i' hiçbir tespitle eşleşmedi (j = -1)
            if (j==-1) tracks_[i].age_ += 1; // Eşleşmeyen iz için yaşı artır
        }
    }

    // 5. Eşleşmeyen izler için:
    //    a. "yaş" veya "kaçırılan" sayısını artır
    //    b. Çok fazla kez kaçırıldıysa, izi sil
    tracks_.erase(
        std::remove_if(
            tracks_.begin(),
            tracks_.end(),
            [](const Track& t) {
                return t.age_ > 10; // 10 kareden fazla eşleşmeyen izleri sil
            }
        ),
        tracks_.end());

    // 6. Eşleşmeyen tespitler için:
    //    a. Yeni Kalman filtreleri (yeni izler) başlat
    for (int i = 0; i < detects.size(); ++i)
    {
        if (!used_detections[i]) // Eğer bu tespit kullanılmadıysa (yani eşleşmediyse)
        {
            tracks_.emplace_back(std::move(detects[i]), ++highest_id_); // Yeni bir iz olarak ekle
        }
    }       

}

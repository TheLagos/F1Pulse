#include "VectorMath.hpp"

#include <cmath>
#include <numeric>

namespace f1_pulse::math {
    float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b)
    {
        if (a.size() != b.size() || a.empty())
        {
            return 0.0f;
        }

        float product = std::inner_product(a.begin(), a.end(), b.begin(), 0.0f);

        float norm_a = std::inner_product(a.begin(), a.end(), a.begin(), 0.0f);
        float norm_b = std::inner_product(b.begin(), b.end(), b.begin(), 0.0f);

        if (norm_a == 0.0f || norm_b == 0.0f)
        {
            return 0.0f;
        }

        return product / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }
} // namespace f1_pulse::math
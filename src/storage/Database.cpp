#include "Database.hpp"
#include "math/VectorMath.hpp"

#include <algorithm>

namespace f1_pulse::storage {
    void Database::add_document(const std::string& id, const std::string& text, const std::vector<float>& embedding)
    {
        m_data.push_back({id, text, embedding});
    }

    std::string Database::query(const std::vector<float>& query_vector)
    {
        if (m_data.empty())
        {
            return "";
        }

        auto match = std::max_element(m_data.begin(), m_data.end(), [&](const DocumentNode& lhs, const DocumentNode& rhs)
        {
            return f1_pulse::math::cosine_similarity(query_vector, lhs.embedding) < f1_pulse::math::cosine_similarity(query_vector, rhs.embedding);
        });

        return match->text;
    }
} // namespace f1_pulse::storage
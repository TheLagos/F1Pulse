#pragma once

#include <vector>
#include <string>

namespace f1_pulse::storage {
    /// @brief 
    struct DocumentNode {
        std::string id;
        std::string text;
        std::vector<float> embedding;
    };

    /// @brief 
    class Database {
        private:
            std::vector<DocumentNode> m_data;
        
        public:
        Database() = default;
        ~Database() = default;

        Database(const Database&) = delete;
        Database& operator=(const Database&) = delete;

        Database(Database&&) noexcept = default;
        Database& operator=(Database&&) noexcept = default;

        /// @brief 
        /// @param id 
        /// @param text 
        /// @param embedding 
        void add_document(const std::string& id, const std::string& text, const std::vector<float>& embedding);

        /// @brief 
        /// @param query_vector 
        /// @return 
        std::string query(const std::vector<float>& query_vector);
    };
} // namespace f1_pulse::storage
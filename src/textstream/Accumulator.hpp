#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <cstddef>

namespace f1_pulse::textstream
{
    /// @brief
    struct PushResult
    {
        std::string ready;
        bool stop_hit{false};
    };

    /// @brief
    class Accumulator
    {
    public:
        Accumulator() = delete;
        explicit Accumulator(const std::vector<std::string> &stop_sequences);

        Accumulator(const Accumulator &) = delete;
        Accumulator &operator=(const Accumulator &) = delete;

        Accumulator(Accumulator &&) = delete;
        Accumulator &operator=(Accumulator &&) = delete;

        ~Accumulator() = default;

        /// @brief
        /// @return
        [[nodiscard]] auto push(std::string_view token) -> PushResult;

        /// @brief
        /// @return
        [[nodiscard]] auto flush() -> std::string;

    private:
        /// @brief
        std::string m_pending_bytes;

        /// @brief
        std::string m_held_text;

        /// @brief
        const std::vector<std::string> &m_stop_sequences;

        /// @brief
        std::size_t m_max_stop_length{0};

        /// @brief
        bool m_closed{false};
    };
} // namespace f1_pulse::textstream
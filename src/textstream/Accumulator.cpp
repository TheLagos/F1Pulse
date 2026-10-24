#include "textstream/Accumulator.hpp"

#include <utility>
#include <algorithm>
#include <optional>

namespace f1_pulse::textstream
{
namespace
{
/// @brief Returns the offset in `text` where the earliest matching stop
/// sequence begins, or std::nullopt if none of `stop_sequences` occur yet.
auto find_stop_sequence(std::string_view text, const std::vector<std::string>& stop_sequences)
    -> std::optional<std::size_t>
{
    std::optional<std::size_t> earliest;

    for (const auto& stop : stop_sequences)
    {
        if (stop.empty())
        {
            continue;
        }

        if (const auto pos = text.find(stop); pos != std::string_view::npos)
        {
            earliest = earliest ? std::min(*earliest, pos) : pos;
        }
    }

    return earliest;
}

/// @brief
auto valid_utf8_prefix_length(std::string_view buffer) -> std::size_t
{
    const std::size_t size = buffer.size();
    const std::size_t limit = std::min<std::size_t>(4, size);

    for (std::size_t back = 1; back <= limit; ++back)
    {
        const std::size_t i = size - back;
        const auto b = static_cast<unsigned char>(buffer[i]);

        if ((b & 0xC0) == 0x80)
        {
            continue;
        }

        std::size_t expected = 0;
        if ((b & 0xF8) == 0xF0)
        {
            expected = 4;
        }
        else if ((b & 0xF0) == 0xE0)
        {
            expected = 3;
        }
        else if ((b & 0xE0) == 0xC0)
        {
            expected = 2;
        }
        else
        {
            expected = 1;
        } // ASCII or invalid lead byte: treat as complete, never stall the stream

        return (back < expected) ? i : size;
    }

    return size;
}
} // namespace

Accumulator::Accumulator(const std::vector<std::string>& stop_sequences) : m_stop_sequences(stop_sequences)
{
    for (const auto& sequence : m_stop_sequences)
    {
        m_max_stop_length = std::max(m_max_stop_length, sequence.size());
    }
}

auto Accumulator::push(std::string_view token) -> PushResult
{
    if (m_closed)
    {
        return {"", false};
    }

    m_pending_bytes.append(token.data(), token.size());
    const std::size_t valid_bytes = valid_utf8_prefix_length(m_pending_bytes);

    m_held_text.append(m_pending_bytes, 0, valid_bytes);
    m_pending_bytes.erase(0, valid_bytes);

    auto stop_pos = find_stop_sequence(m_held_text, m_stop_sequences);

    if (stop_pos.has_value())
    {
        m_closed = true;
        std::string ready = m_held_text.substr(0, stop_pos.value());

        m_pending_bytes.clear();
        m_held_text.clear();

        return {std::move(ready), true};
    }

    if (m_max_stop_length <= 1)
    {
        std::string ready(std::move(m_held_text));
        m_held_text.clear();

        return {std::move(ready), false};
    }

    const std::size_t keep = m_max_stop_length - 1;
    std::size_t cut = (m_held_text.size() > keep) ? m_held_text.size() - keep : 0;

    while (cut > 0 && (static_cast<unsigned char>(m_held_text[cut]) & 0xC0) == 0x80)
    {
        --cut;
    }

    std::string ready = m_held_text.substr(0, cut);
    m_held_text.erase(0, cut);
    return {std::move(ready), false};
}

auto Accumulator::flush() -> std::string
{
    std::string ready = std::move(m_held_text);

    m_held_text.clear();
    m_pending_bytes.clear();
    m_closed = true;

    return ready;
}

} // namespace f1_pulse::textstream
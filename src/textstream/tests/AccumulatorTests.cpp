#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "textstream/Accumulator.hpp"

#include <string>
#include <vector>

using f1_pulse::textstream::Accumulator;

// Accumulator stores a reference to the stop sequences: every test keeps them
// in a local vector that outlives the Accumulator.

TEST_CASE("no stop sequences: split Cyrillic character is held until complete")
{
    const std::vector<std::string> stops;
    Accumulator acc(stops);

    const auto first = acc.push("\xD1");
    CHECK(first.ready.empty());
    CHECK_FALSE(first.stop_hit);

    const auto second = acc.push("\x8F");
    CHECK(second.ready == "\xD1\x8F"); // U+044F CYRILLIC SMALL LETTER YA
    CHECK_FALSE(second.stop_hit);
}

TEST_CASE("empty token returns empty ready and no stop")
{
    const std::vector<std::string> stops{"STOP"};
    Accumulator acc(stops);

    const auto result = acc.push("");
    CHECK(result.ready.empty());
    CHECK_FALSE(result.stop_hit);
}

TEST_CASE("stop sequence fully inside one token")
{
    const std::vector<std::string> stops{"STOP"};
    Accumulator acc(stops);

    const auto result = acc.push("Hello STOP world");
    CHECK(result.ready == "Hello ");
    CHECK(result.stop_hit);
}

TEST_CASE("stop sequence split across two tokens is detected and does not leak")
{
    const std::vector<std::string> stops{"</s>"};
    Accumulator acc(stops);

    const auto first = acc.push("abc</");
    CHECK_FALSE(first.stop_hit);
    CHECK(first.ready.find('<') == std::string::npos);

    const auto second = acc.push("s>xyz");
    CHECK(second.stop_hit);

    const std::string delivered = first.ready + second.ready;
    CHECK(delivered == "abc");
}

TEST_CASE("tail that looks like a stop prefix is held back, then returned by flush")
{
    const std::vector<std::string> stops{"END"};
    Accumulator acc(stops);

    const auto result = acc.push("hello EN");
    CHECK(result.ready == "hello ");
    CHECK_FALSE(result.stop_hit);

    CHECK(acc.flush() == "EN");
}

TEST_CASE("flush after a caught stop returns an empty string")
{
    const std::vector<std::string> stops{"STOP"};
    Accumulator acc(stops);

    const auto result = acc.push("before STOP after");
    REQUIRE(result.stop_hit);

    CHECK(acc.flush().empty());
}

TEST_CASE("push after flush returns empty ready and no stop")
{
    const std::vector<std::string> stops{"STOP"};
    Accumulator acc(stops);

    const auto pushed = acc.push("abc");
    CHECK_FALSE(pushed.stop_hit);
    (void)acc.flush();

    const auto result = acc.push("more text STOP");
    CHECK(result.ready.empty());
    CHECK_FALSE(result.stop_hit);
}

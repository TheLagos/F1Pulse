add_executable(f1_textstream_tests AccumulatorTests.cpp)

target_compile_features(f1_textstream_tests PRIVATE cxx_std_23)
target_link_libraries(f1_textstream_tests PRIVATE f1::textstream doctest::doctest f1_warnings)

add_test(NAME textstream.accumulator COMMAND f1_textstream_tests)

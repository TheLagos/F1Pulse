f1_add_module(ai
    SOURCES LlamaEngine.cpp
    PUBLIC_DEPS llama
    PRIVATE_DEPS f1::textstream)

# Database.cpp calls f1_pulse::math::cosine_similarity.
f1_add_module(storage
    SOURCES Database.cpp
    PRIVATE_DEPS f1::math)

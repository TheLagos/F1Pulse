FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=nointeractive

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    cmake \
    clangd \
    clang-format \
    valgrind \
    ccache \
    ninja-build \
    gdb \
    qt6-base-dev \
    qt6-declarative-dev \
    libgl1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*
    
WORKDIR /workspace
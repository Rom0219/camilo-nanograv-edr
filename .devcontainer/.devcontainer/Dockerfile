FROM mcr.microsoft.com/devcontainers/miniconda:latest

RUN apt-get update && apt-get install -y \
  build-essential \
  git \
  wget \
  automake \
  autoconf \
  libtool \
  && rm -rf /var/lib/apt/lists/*

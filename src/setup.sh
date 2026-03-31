#!/bin/bash

# 1. Install Rust (non-interactive mode)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# 2. Update package list and install CMake
apt update && apt install cmake -y

# 3. Install Clang
apt-get install clang -y

# 4. Set environment variables for the current session
export CC=clang
export CXX=clang++

# 5. Persist environment variables to .bashrc for future sessions
echo 'export CC=clang' >> ~/.bashrc
echo 'export CXX=clang++' >> ~/.bashrc

# 6. Source the Rust environment and bashrc
source $HOME/.cargo/env
source ~/.bashrc

# 7. Install Vim
apt-get install vim -y

echo "Setup complete! Rust, CMake, Clang, and Vim are installed."

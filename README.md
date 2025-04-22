# Baba Is RL

Reinforcement Learning for [Baba Is You](https://store.steampowered.com/app/736260/Baba_Is_You/)

The simulator is taken from [baba-is-auto](https://github.com/utilForever/baba-is-auto/)

## baba-is-auto
baba-is-auto is Baba Is You simulator using C++ with some reinforcement learning. The code is built on C++17 and can be compiled with commonly available compilers such as g++, clang++, or Microsoft Visual Studio. baba-is-auto currently supports macOS (10.14 or later), Ubuntu (18.04 or later), Windows (Visual Studio 2017 or later), and Windows Subsystem for Linux (WSL). Other untested platforms that support C++17 also should be able to build baba-is-auto.

## What is "Baba Is You"?

[Baba Is You](https://hempuli.com/baba/) is an award-winning puzzle game where you can change the rules by which you play. In every level, the rules themselves are present as blocks you can interact with; by manipulating them, you can change how the level works and cause surprising, unexpected interactions! With some simple block-pushing you can turn yourself into a rock, turn patches of grass into dangerously hot obstacles, and even change the goal you need to reach to something entirely different.

## Quick Start
You will need CMake to build the code. If you're using Windows, you need Visual Studio 2017 in addition to CMake.

First, clone the code:
```
git clone https://github.com/baba-is-rl/baba-is-rl.git --recursive
cd baba-is-rl
```

### C++ API

For macOS or Linux or Windows Subsystem for Linux (WSL):

```
mkdir build
cd build
cmake ..
make
```

For Windows:

```
mkdir build
cd build
cmake .. -G"Visual Studio 15 2017 Win64"
MSBuild baba-is-rl.sln /p:Configuration=Release
```

### Python API

Build and install the package by running

```
pip install -U .
```

### Reinforcement Learning
The RL algorithms are implemented in [Extensions/BabaRL/](./Extensions/BabaRL).

We recommend using [`uv`](https://github.com/astral-sh/uv)  for running the algorithms / installing deps.

e.g: `uv run DQN.py` will run the DQN algorithm inside [Extensions/BabaRL/src/babaisyou_v0](./Extensions/BabaRL/src/babaisyou_v0)

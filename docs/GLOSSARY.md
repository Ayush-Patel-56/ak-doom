# Glossary

Simple meanings of the words used in these docs.

| Word | Simple meaning |
|------|----------------|
| **Agent** | The AI player. |
| **Observation** | What the agent receives from the game at each step (rays, enemies, health...). |
| **Action** | The buttons the agent presses (move, turn, shoot, use). |
| **Tick / tic** | One tiny game time step. DOOM runs 35 ticks per second. We decide every 4 ticks. |
| **Map unit** | DOOM's distance unit. The player is about 32 units wide and 56 tall; rays reach 1,024 units. |
| **`doom_env` / `make_env`** | The official wrapper that turns DOOM into a Gymnasium environment with the fixed observation format. |
| **`privileged_state()`** | A wrapper helper that gives the true position, kills and damage. Training only. |
| **MultiBinary(9)** | The action format: 9 buttons, each on (1) or off (0). |
| **uv** | A fast Python package manager the template uses (`uv sync`, `uv run`). |
| **Validator** | `validate.py`, the official pre-submission check. |
| **Episode** | One attempt at one map, from start until exit, death or time limit. |
| **Reinforcement Learning (RL)** | Learning by trial and error: good results give rewards, bad results give penalties. |
| **Reward** | A number telling the agent "that was good" (+) or "that was bad" (−). |
| **Reward shaping** | Adding extra small rewards to guide learning (e.g. for visiting new places). |
| **Policy** | The part of the brain that chooses actions. |
| **Critic / value** | The part that guesses "how good is my situation?", used only to help learning. |
| **PPO** | Proximal Policy Optimization. A popular, stable RL learning method that takes small, safe update steps. |
| **GAE** | A trick to estimate how good each action really was, mixing short- and long-term rewards. |
| **Encoder** | A small network that turns raw input into a compact list of numbers. |
| **Embedding** | That compact list of numbers. |
| **Ray / raycast** | An invisible line from the player outward that reports what it hits and how far away it is. |
| **Ego-centric** | "From my own point of view": left/right/front relative to the player, not the world. |
| **Depth buffer** | An image where each pixel holds a distance instead of a colour. |
| **Vitals** | Health, armor, ammo, keycards and speed. |
| **Attention** | A network trick that decides which items in a list matter most right now. |
| **GRU / LSTM** | Recurrent networks: short-term memory that carries information from one step to the next. |
| **Odometer / dead reckoning** | Estimating where you are by adding up your own movements. |
| **Mind map (occupancy grid)** | A grid sketch of the area: which cells are walls, open space, doors or visited. |
| **Landmark notebook** | Our memory slots for important objects like keycards, locked doors and the exit. |
| **Manager / worker (hierarchical policy)** | The manager picks a goal; the worker presses buttons to achieve it. |
| **Curiosity / intrinsic reward** | Extra reward for seeing new things, so the agent explores. |
| **RND** | Random Network Distillation, a curiosity method: "if I can't predict it, it's new". |
| **Curriculum learning** | Teaching easy tasks first, then harder ones. |
| **Domain randomisation** | Changing textures, spawns and monsters during training so the agent doesn't depend on one look. |
| **Generalisation** | Doing well on maps never seen during training. |
| **Overfitting** | Doing well only on training maps because they were memorised. |
| **Zero-shot** | Playing a brand-new map with no extra training on it. |
| **Partial observability** | The agent can't see everything, only what's in view. |
| **Ablation** | Removing one part to prove how much it helps. |
| **Checkpoint** | A saved copy of the brain at some point in training. |
| **Champion** | The best checkpoint so far, the one we submit. |
| **Mini-League** | Our own test tournament on held-out maps that copies the official scoring. |
| **Held-out maps** | Maps kept aside and never used for training. |
| **Asymmetric critic** | A critic that sees extra (privileged) information during training only. |
| **Teacher–student** | A teacher with extra information trains first; the student (our real agent) learns to copy it. |
| **Privileged information** | Information the real agent will never get (e.g. true position), used only in training. |
| **WAD** | A DOOM map/data file. |
| **IWAD** | The main game data file (here: Freedoom 2). |
| **UDMF** | A modern text-based DOOM map format. |
| **OBLIGE / ObAddon** | Programs that generate random, playable DOOM levels. |
| **Freedoom** | A free, open-source replacement for the original DOOM game data. |
| **ViZDoom** | The research library that lets Python control DOOM. |
| **Gymnasium** | The standard Python interface for RL environments (`reset`, `step`). |
| **Git LFS** | Git Large File Storage, for storing big files like model weights. |
| **Latency** | How long one decision takes. |
| **TorchScript** | A way to make a PyTorch model run faster without Python overhead. |

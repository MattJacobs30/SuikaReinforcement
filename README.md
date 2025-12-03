# Suika Game Reinforcement Learning

This project implements a Reinforcement Learning (RL) agent to play the Suika Game (Watermelon Game) using Deep Q-Networks (DQN).

## Quick Start

### 1. Install Dependencies
Make sure you have Python 3.10 environment. Then install the required packages:

```bash
pip install -r requirements.txt
```

### 2. Run the Trained Agent
We have provided a pre-trained model: `suika_dqn_mlp_final.zip`. To watch it play:

```bash
python rl_env/test_model.py --model suika_dqn_mlp_final.zip
```

**Options:**
- `--episodes N`: Run for N episodes (default: 1).
- `--fps N`: Limit playback speed to N FPS (default: 60).
- `--stochastic`: Use random actions based on probabilities (default: deterministic/best action).
- `--empty`: Start with an empty board (no initial random fruits).

Example:
```bash
python rl_env/test_model.py --model suika_dqn_mlp_final.zip --episodes 3 --fps 120
```

## Other Ways to Run

### Play the Game Yourself
If you want to play the game manually using your mouse:

```bash
python rl_env/human_play.py
```
- **Controls**: Move mouse to position the cloud, click to drop the fruit.

### Train a New Agent
To train a fresh agent or continue training:

```bash
python rl_env/train.py
```
This will save checkpoints to `models_dqn/` and logs to `logs_dqn/` in the current directory.

## File Overview

- **`suika_dqn_mlp_final.zip`**: The final trained DQN model ready for testing.
- **`rl_env/`**: Contains the Reinforcement Learning scripts.
  - `test_model.py`: Script to load and watch a trained model play.
  - `train.py`: Script to train the DQN agent.
  - `human_play.py`: Script for human gameplay.
  - `suika_env.py`: The Gymnasium environment wrapper for the game.
- **`suika/`**: Contains the core game logic and assets. Taken from an open source project seen here: https://github.com/Ole-Batting/suika
- **`requirements.txt`**: List of Python dependencies.


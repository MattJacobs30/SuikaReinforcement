# Suika RL Environment

This directory contains a Reinforcement Learning environment for the Suika game (from `../suika/part2`), compatible with Gymnasium and Stable Baselines 3.

## Structure

- `suika_env.py`: The custom Gymnasium environment `SuikaEnv`.
- `train.py`: A script to train a DQN agent using Stable Baselines 3.
- `human_play.py`: A script to play the game manually to test the environment and game over logic.
- `test_env.py`: A simple script to run the environment with a random agent.

## Prerequisites

Ensure you have the requirements installed:
```bash
pip install -r ../requirements.txt
```

## Usage

### Training a DQN Agent

To start training a DQN agent:

```bash
python train.py
```

This will save logs to `logs_dqn/` and models to `models_dqn/`.
Note: The environment is configured to resize observations to 84x84 and use discrete actions (50 bins) for DQN compatibility.

### Testing as a Human

To play the game yourself and verify the environment logic (e.g., Game Over):

```bash
python human_play.py
```
- Use your **Mouse** to move the cloud.
- **Click** to drop the fruit.
- Press **ESC** to quit.

### Watching a Random Agent

To see the environment in action with a random agent:

```bash
python test_env.py
```

## Environment Details

- **Action Space**: 
    - `Discrete(50)` (default for DQN training): Bins mapped to x-positions.
    - `Box(-1, 1)` (optional, used for human play): Continuous x-position.
- **Observation Space**: `Box(84, 84, 3)` RGB image of the game screen (resized from 720p).
- **Reward**: The increase in score achieved in the step (merging fruits).

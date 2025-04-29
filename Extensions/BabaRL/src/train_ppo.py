import argparse
import os
import time
from collections import deque
from glob import glob

import gym
import numpy as np
import pyBaba
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.distributions import Categorical

from environment import (
    register_env,
)
from ppo_model import ActorCritic

LEARNING_RATE = 3e-4
GAMMA = 0.99  # Discount factor
LAMBDA = 0.95  # GAE lambda
CLIP_EPS = 0.2  # PPO clipping epsilon
ENTROPY_COEF = 0.01  # Entropy coefficient
VALUE_LOSS_COEF = 0.5  # Value loss coefficient
EPOCHS = 10  # PPO epochs per update
MINI_BATCH_SIZE = 64
NUM_STEPS = 2048  # Steps per policy update (collect N steps)
MAX_TRAIN_STEPS = 10_000_000  # Total training steps
ANNEAL_LR = True  # Whether to linearly anneal learning rate

MAPS_DIR = "../../../Resources/Maps"
SPRITES_PATH = "../sprites"
CHECKPOINT_DIR = "./checkpoints"
LOG_DIR = "./logs"

# Curriculum
LEVELS = [
    *sorted(glob("priming/*txt", root_dir=MAPS_DIR)),
    "baba_is_you.txt",
    "out_of_reach.txt",
    # "off_limits.txt",
]

# Curriculum parameters
CURRICULUM_THRESHOLD = 0.85  # Avg win rate to advance (adjust)
CURRICULUM_WINDOW = 50  # Episodes to average for threshold


def compute_gae(next_value, rewards, dones, values, gamma, lambda_):
    """Computes Generalized Advantage Estimation."""
    advantages = torch.zeros_like(rewards)
    last_gae_lam = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            nextnonterminal = 1.0 - dones[-1]  # Use final done state
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1]  # Use next step's done state
            nextvalues = values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = last_gae_lam = (
            delta + gamma * lambda_ * nextnonterminal * last_gae_lam
        )
    return advantages


def dict_to_torch(obs_dict, device):
    """Converts observation dictionary numpy arrays to torch tensors."""
    grid = torch.tensor(obs_dict["grid"], dtype=torch.float32).to(device)
    rules = torch.tensor(obs_dict["rules"], dtype=torch.long).to(device)
    rule_mask = torch.tensor(obs_dict["rule_mask"], dtype=torch.long).to(device)
    return grid, rules, rule_mask


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--cuda", "-c", action="store_true", help="Use Cuda")
    parser.add_argument(
        "--run-name",
        "-r",
        type=str,
        default=f"ppo_baba_{int(time.time())}",
        help="Run name for logs/checkpoints",
    )
    parser.add_argument(
        "--load-checkpoint",
        "-l",
        type=str,
        help="path to checkpoint file to load state from",
    )
    parser.add_argument(
        "--disable-render",
        "-d",
        action="store_true",
        help="disable rendering while training",
    )
    parser.add_argument(
        "--save-after",
        "-s",
        type=int,
        default=50,
        metavar="N",
        help="save model state after %(metavar)s episodes",
    )
    parser.add_argument(
        "--max-steps",
        "-m",
        type=int,
        default=200,
        metavar="N",
        help="max number of steps per episode",
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()

    print("Called With:", args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(args.seed)
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    run_log_dir = os.path.join(LOG_DIR, args.run_name)
    run_checkpoint_dir = os.path.join(CHECKPOINT_DIR, args.run_name)
    os.makedirs(run_log_dir, exist_ok=True)
    os.makedirs(run_checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(run_log_dir)

    current_curriculum_stage = 0
    current_map_name = LEVELS[current_curriculum_stage]
    env_id = f"baba-{os.path.splitext(current_map_name)[0]}-v0"
    register_env(
        env_id,
        current_map_name,
        os.path.splitext(current_map_name)[0],
        MAPS_DIR,
        SPRITES_PATH,
        enable_render=not args.disable_render,
    )
    env = gym.make(env_id)
    env.seed(args.seed)

    grid_shape = env.observation_space["grid"].shape
    num_actions = env.action_space.n

    agent = ActorCritic(grid_shape, num_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)

    if args.load_checkpoint:
        if os.path.isfile(args.load_checkpoint):
            print(
                f"Loading agent/optimizer weights from checkpoint: {args.load_checkpoint}"
            )
            try:
                checkpoint = torch.load(args.load_checkpoint, map_location=device)

                agent.load_state_dict(checkpoint["agent_state_dict"])
                print("-> Agent weights loaded successfully.")

                if "optimizer_state_dict" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    print("-> Optimizer state loaded successfully.")
                else:
                    print(
                        "-> Optimizer state not found in checkpoint, initializing fresh."
                    )

            except Exception as e:
                print(
                    f"ERROR loading checkpoint: {e}. Check file integrity and model compatibility."
                )
                print("Starting training from scratch.")
                agent = ActorCritic(grid_shape, num_actions).to(device)
                optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)

        else:
            print(
                f"Warning: Checkpoint file not found at {args.load_checkpoint}. Starting from scratch."
            )
    else:
        print("No checkpoint specified, starting training from scratch.")

    # Pre-allocate storage tensors on the correct device
    grids = torch.zeros((NUM_STEPS,) + grid_shape).to(device)
    rules_storage = torch.zeros(
        (NUM_STEPS,) + env.observation_space["rules"].shape, dtype=torch.long
    ).to(device)
    rule_masks = torch.zeros(
        (NUM_STEPS,) + env.observation_space["rule_mask"].shape, dtype=torch.long
    ).to(device)
    actions = torch.zeros(NUM_STEPS, dtype=torch.long).to(device)
    log_probs = torch.zeros(NUM_STEPS).to(device)
    rewards = torch.zeros(NUM_STEPS).to(device)
    dones = torch.zeros(NUM_STEPS).to(device)
    values = torch.zeros(NUM_STEPS).to(device)

    global_step = 0
    start_time = time.time()
    next_obs_dict = env.reset()

    # print(f"*** Initial State Check ({current_map_name}) ***")
    initial_play_state = env.game.GetPlayState()
    initial_player_icon = env.game.GetPlayerIcon()
    initial_rules = pyBaba.Preprocess.GetAllRules(env.game)
    # print(f"Initial Play State: {initial_play_state.name}")
    # print(f"Initial Player Icon: {initial_player_icon.name}")
    # print(f"Initial Rule Count: {len(initial_rules)}")
    if initial_play_state == pyBaba.PlayState.LOST:
        print("!!!!!! WARNING: Game starts in LOST state !!!!!!")
    if initial_player_icon == pyBaba.ObjectType.ICON_EMPTY:
        print("!!!!!! WARNING: No initial Player Icon found !!!!!!")

    next_grid, next_rules, next_rule_mask = dict_to_torch(next_obs_dict, device)
    next_done = torch.zeros(1).to(device)

    episode_rewards = deque(maxlen=CURRICULUM_WINDOW)
    episode_wins = deque(maxlen=CURRICULUM_WINDOW)  # 1 if win, 0 otherwise
    num_updates = MAX_TRAIN_STEPS // NUM_STEPS

    for update in range(1, num_updates + 1):
        if ANNEAL_LR:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * LEARNING_RATE
            optimizer.param_groups[0]["lr"] = lrnow

        current_episode_reward = 0
        current_episode_steps = 0

        # Rollout Phase
        agent.eval()
        for step in range(NUM_STEPS):
            global_step += 1
            current_episode_steps += 1

            grids[step] = next_grid
            rules_storage[step] = next_rules
            rule_masks[step] = next_rule_mask
            dones[step] = next_done.item()

            with torch.no_grad():
                logits, value = agent(
                    next_grid.unsqueeze(0),
                    next_rules.unsqueeze(0),
                    next_rule_mask.unsqueeze(0),
                )
                values[step] = value.flatten()

            probs = Categorical(logits=logits)
            action = probs.sample()
            actions[step] = action.item()
            log_probs[step] = probs.log_prob(action)

            # Environment Step
            obs_dict, reward, done, info = env.step(action.item())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            current_episode_reward += reward

            env.render()

            next_obs_dict = obs_dict
            next_grid, next_rules, next_rule_mask = dict_to_torch(obs_dict, device)
            next_done = torch.tensor(done, dtype=torch.float32).to(device)

            if done:
                print(
                    f"Global Step: {global_step}, Stage: {current_curriculum_stage} ({current_map_name}), Episode Reward: {current_episode_reward:.2f}, Steps: {current_episode_steps}"
                )
                writer.add_scalar(
                    "charts/episode_reward", current_episode_reward, global_step
                )
                writer.add_scalar(
                    "charts/episode_length", current_episode_steps, global_step
                )
                episode_rewards.append(current_episode_reward)
                is_win = 1.0 if reward >= 1.0 else 0.0
                episode_wins.append(is_win)
                writer.add_scalar(
                    "charts/win_rate_window", np.mean(episode_wins), global_step
                )
                writer.add_scalar(
                    "charts/avg_reward_window", np.mean(episode_rewards), global_step
                )
                writer.add_scalar(
                    f"curriculum/stage_{current_curriculum_stage}_reward",
                    current_episode_reward,
                    global_step,
                )
                writer.add_scalar(
                    f"curriculum/stage_{current_curriculum_stage}_win",
                    is_win,
                    global_step,
                )

                # Curriculum Advancement
                needs_reset = True
                if (
                    len(episode_wins) == CURRICULUM_WINDOW
                    and np.mean(episode_wins) >= CURRICULUM_THRESHOLD
                ):
                    if current_curriculum_stage < len(LEVELS) - 1:
                        current_curriculum_stage += 1
                        current_map_name = LEVELS[current_curriculum_stage]
                        new_map_path_for_reset = os.path.join(
                            MAPS_DIR, current_map_name
                        )
                        print(
                            f"\n--- Advancing to Curriculum Stage {current_curriculum_stage}: {current_map_name} ---"
                        )
                        writer.add_text(
                            "curriculum/current_level", current_map_name, global_step
                        )
                        episode_rewards.clear()
                        episode_wins.clear()
                        # Update env_id for registration
                        env_id = f"baba-{current_curriculum_stage}-v0"
                        register_env(
                            env_id,
                            current_map_name,
                            os.path.splitext(current_map_name)[0],
                            MAPS_DIR,
                            SPRITES_PATH,
                            enable_render=not args.disable_render,
                        )
                        env.close()
                        env = gym.make(env_id)
                        env.seed(args.seed)
                        next_obs_dict = env.reset()
                        needs_reset = False

                    else:
                        new_map_path_for_reset = None
                else:
                    # No advancement, reset with current map
                    new_map_path_for_reset = None

                if needs_reset:
                    # print(f"Resetting map: {current_map_name}")
                    next_obs_dict = env.reset(new_map_path=new_map_path_for_reset)

                # print(
                #     f"*** State after reset for {current_map_name} (Global Step {global_step}) ***"
                # )
                reset_play_state = env.game.GetPlayState()
                reset_player_icon = env.game.GetPlayerIcon()
                reset_rules = pyBaba.Preprocess.GetAllRules(env.game)
                # print(f"Reset Play State: {reset_play_state.name}")
                # print(f"Reset Player Icon: {reset_player_icon.name}")
                # print(f"Reset Rule Count: {len(reset_rules)}")

                if reset_play_state == pyBaba.PlayState.LOST:
                    print(
                        "!!!!!! WARNING: Game is in LOST state immediately after reset !!!!!!"
                    )
                if reset_player_icon == pyBaba.ObjectType.ICON_EMPTY:
                    print(
                        "!!!!!! WARNING: No Player Icon found immediately after reset !!!!!!"
                    )

                # Check if BABA IS YOU is present (or equivalent)
                has_you_rule = False
                for rule in reset_rules:
                    try:
                        r_objs = rule.GetObjects()
                        if (
                            len(r_objs) == 3
                            and r_objs[1].HasType(pyBaba.ObjectType.IS)
                            and r_objs[2].HasType(pyBaba.ObjectType.YOU)
                        ):
                            has_you_rule = True
                            # print(f"  Found YOU rule: {r_objs[0].GetTypes()[0].name} IS YOU")
                            break
                    except Exception as e:
                        print(f"had exception @ N is YOU rule check: {e}")
                        pass  # Ignore potential errors getting types if rule is malformed
                if not has_you_rule and reset_play_state != pyBaba.PlayState.WON:
                    print("!!!!!! WARNING: No 'X IS YOU' rule found after reset !!!!!!")

                next_grid, next_rules, next_rule_mask = dict_to_torch(
                    next_obs_dict, device
                )
                next_done = torch.zeros(1).to(device)
                current_episode_reward = 0
                current_episode_steps = 0

        # Calculate Advantages and Returns
        agent.eval()
        with torch.no_grad():
            _, next_value = agent(
                next_grid.unsqueeze(0),
                next_rules.unsqueeze(0),
                next_rule_mask.unsqueeze(0),
            )
            advantages = compute_gae(
                next_value.flatten(), rewards, dones, values, GAMMA, LAMBDA
            )
            returns = advantages + values

        # PPO Update Phase
        agent.train()
        b_inds = np.arange(NUM_STEPS)
        for epoch in range(EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, NUM_STEPS, MINI_BATCH_SIZE):
                end = start + MINI_BATCH_SIZE
                mb_inds = b_inds[start:end]

                # Get data for minibatch
                mb_grids = grids[mb_inds]
                mb_rules = rules_storage[mb_inds]
                mb_masks = rule_masks[mb_inds]
                mb_actions = actions[mb_inds]
                mb_log_probs = log_probs[mb_inds]
                mb_advantages = advantages[mb_inds]
                mb_returns = returns[mb_inds]

                new_logits, new_value = agent(mb_grids, mb_rules, mb_masks)
                new_probs = Categorical(logits=new_logits)
                new_log_prob = new_probs.log_prob(mb_actions)
                entropy = new_probs.entropy().mean()

                # Policy Loss (Clipped)
                log_ratio = new_log_prob - mb_log_probs
                ratio = torch.exp(log_ratio)
                clipped_ratio = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS)
                policy_loss1 = -mb_advantages * ratio
                policy_loss2 = -mb_advantages * clipped_ratio
                policy_loss = torch.max(policy_loss1, policy_loss2).mean()

                # Value Loss (Clipped or MSE)
                new_value = new_value.view(-1)
                # value_loss_unclipped = (new_value - mb_returns) ** 2
                # value_clipped = values[mb_inds] + torch.clamp(new_value - values[mb_inds], -CLIP_EPS, CLIP_EPS)
                # value_loss_clipped = (value_clipped - mb_returns) ** 2
                # value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                value_loss = 0.5 * ((new_value - mb_returns) ** 2).mean()

                # Total Loss
                loss = (
                    policy_loss - ENTROPY_COEF * entropy + VALUE_LOSS_COEF * value_loss
                )

                # Optimization
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)  # Gradient clipping
                optimizer.step()

        writer.add_scalar("losses/total_loss", loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
        writer.add_scalar("losses/value_loss", value_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy.item(), global_step)
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar(
            "charts/SPS", int(NUM_STEPS / (time.time() - start_time)), global_step
        )
        print(
            f"Update {update}/{num_updates}, SPS: {int(NUM_STEPS / (time.time() - start_time))}"
        )
        start_time = time.time()

        if update % args.save_after == 0:  # Save every 20 updates
            checkpoint_path = os.path.join(
                run_checkpoint_dir, f"ppo_baba_update_{update}.pth"
            )
            torch.save(
                {
                    "agent_state_dict": agent.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "global_step": global_step,
                    "curriculum_stage": current_curriculum_stage,
                    "episode_rewards": list(episode_rewards),
                    "episode_wins": list(episode_wins),
                },
                checkpoint_path,
            )
            print(f"Checkpoint saved to {checkpoint_path}")

    env.close()
    writer.close()
    print("Training finished.")

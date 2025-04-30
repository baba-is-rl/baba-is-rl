import argparse
import os
import time
from collections import deque

import gym
import numpy as np
import pyBaba
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.distributions import Categorical

from environment import register_env
from hrl_model import HierarchicalActorCritic
from constants import (
    MAX_MAP_HEIGHT, MAX_MAP_WIDTH, MAX_RULES, NUM_OBJECT_TYPES, PAD_INDEX,
    MANAGER_UPDATE_FREQ_K, NUM_OPTIONS
)

LEARNING_RATE = 3e-4
GAMMA = 0.95  
LAMBDA = 0.95 
CLIP_EPS = 0.2  
ENTROPY_COEF_WORKER = 0.01
ENTROPY_COEF_MANAGER = 0.05 
VALUE_LOSS_COEF = 0.5  
EPOCHS = 10  # PPO epochs per update
MINI_BATCH_SIZE = 64
NUM_STEPS = 2048  # Worker steps per policy update
MAX_TRAIN_STEPS = 10_000_000  # Total limit on training steps
ANNEAL_LR = True  # Whether to linearly anneal learning rate


INITIAL_MAP = "priming/lvl1.txt"
MAPS_DIR = "../../../../Resources/Maps"
SPRITES_PATH = "../../sprites"
CHECKPOINT_DIR = "./checkpoints_hrl"
LOG_DIR = "./logs_hrl"

def compute_gae(next_value, rewards, dones, values, gamma, lambda_):
    advantages = torch.zeros_like(rewards)
    last_gae_lam = 0
    num_steps = len(rewards)
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            nextnonterminal = 1.0 - dones[-1] 
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t+1]
            nextvalues = values[t+1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = last_gae_lam = delta + gamma * lambda_ * nextnonterminal * last_gae_lam
    return advantages

def dict_to_torch(obs_dict, device):
    grid = torch.tensor(obs_dict["grid"], dtype=torch.float32).to(device)
    rules = torch.tensor(obs_dict["rules"], dtype=torch.long).to(device)
    rule_mask = torch.tensor(obs_dict["rule_mask"], dtype=torch.long).to(device)
    return grid, rules, rule_mask

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--cuda", action="store_true", help="Use Cuda")
    parser.add_argument(
        "--run_name",
        type=str,
        default=f"hrl_baba_{int(time.time())}",
        help="Run name for logs/checkpoints",
    )
    parser.add_argument(
        "--load-checkpoint",
        "-l",
        type=str,
        help="path to checkpoint file to load state from",
    )
    parser.add_argument(
        "--k", type=int, default=MANAGER_UPDATE_FREQ_K, help="Manager update frequency"
    )
    parser.add_argument(
        "--num-options", type=int, default=NUM_OPTIONS, help="Number of options"
    )
    args = parser.parse_args()

    MANAGER_UPDATE_FREQ_K = args.k
    NUM_OPTIONS = args.num_options

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(args.seed)
        print(f"Using CUDA - K={MANAGER_UPDATE_FREQ_K}, Options={NUM_OPTIONS}")
    else:
        device = torch.device("cpu")
        print(f"Using CPU - K={MANAGER_UPDATE_FREQ_K}, Options={NUM_OPTIONS}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    run_log_dir = os.path.join(LOG_DIR, args.run_name)
    run_checkpoint_dir = os.path.join(CHECKPOINT_DIR, args.run_name)
    os.makedirs(run_log_dir, exist_ok=True)
    os.makedirs(run_checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(run_log_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    writer.add_scalar("params/k", MANAGER_UPDATE_FREQ_K)
    writer.add_scalar("params/num_options", NUM_OPTIONS)


    env_id = f"baba-{os.path.splitext(INITIAL_MAP)[0]}-hrl-v0"
    register_env(
        env_id,
        INITIAL_MAP,
        os.path.splitext(INITIAL_MAP)[0] + " (HRL)",
        MAPS_DIR,
        SPRITES_PATH,
    )
    env = gym.make(env_id)
    env.seed(args.seed)

    grid_shape = env.observation_space["grid"].shape
    num_primitive_actions = env.action_space.n

    agent = HierarchicalActorCritic(
        grid_shape, num_primitive_actions, NUM_OPTIONS
    ).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)

    global_step = 0
    start_update = 1

    if args.load_checkpoint:
        if os.path.isfile(args.load_checkpoint):
            print(f"Loading checkpoint: {args.load_checkpoint}")
            try:
                checkpoint = torch.load(args.load_checkpoint, map_location=device)
                agent.load_state_dict(checkpoint["agent_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                global_step = checkpoint.get("global_step", 0)
                start_update = (global_step // NUM_STEPS) + 1
                print(f"-> Resuming from Global Step: {global_step}, Update: {start_update}")
            except Exception as e:
                print(f"ERROR loading checkpoint: {e}. Starting fresh.")
                global_step = 0
                start_update = 1
        else:
            print(f"Warning: Checkpoint file not found: {args.load_checkpoint}. Starting fresh.")
    else:
        print("Starting training from scratch.")


    worker_grids = torch.zeros((NUM_STEPS,) + grid_shape).to(device)
    worker_rules = torch.zeros(
        (NUM_STEPS,) + env.observation_space["rules"].shape, dtype=torch.long
    ).to(device)
    worker_rule_masks = torch.zeros(
        (NUM_STEPS,) + env.observation_space["rule_mask"].shape, dtype=torch.long
    ).to(device)
    worker_actions = torch.zeros(NUM_STEPS, dtype=torch.long).to(device) 
    worker_options = torch.zeros(NUM_STEPS, dtype=torch.long).to(device) 
    worker_log_probs = torch.zeros(NUM_STEPS).to(device)
    worker_rewards = torch.zeros(NUM_STEPS).to(device)
    worker_dones = torch.zeros(NUM_STEPS).to(device)
    worker_values = torch.zeros(NUM_STEPS).to(device)

    manager_steps_per_update = NUM_STEPS // MANAGER_UPDATE_FREQ_K + 1
    manager_grids = torch.zeros((manager_steps_per_update,) + grid_shape).to(device)
    manager_rules = torch.zeros(
        (manager_steps_per_update,) + env.observation_space["rules"].shape, dtype=torch.long
    ).to(device)
    manager_rule_masks = torch.zeros(
        (manager_steps_per_update,) + env.observation_space["rule_mask"].shape, dtype=torch.long
    ).to(device)
    manager_options_chosen = torch.zeros(manager_steps_per_update, dtype=torch.long).to(device) 
    manager_log_probs = torch.zeros(manager_steps_per_update).to(device)
    manager_rewards = torch.zeros(manager_steps_per_update).to(device) 
    manager_dones = torch.zeros(manager_steps_per_update).to(device)  
    manager_values = torch.zeros(manager_steps_per_update).to(device)

    start_time = time.time()
    next_obs_dict = env.reset()
    next_grid, next_rules, next_rule_mask = dict_to_torch(next_obs_dict, device)
    next_worker_done = torch.zeros(1).to(device)
    next_manager_option = torch.randint(0, NUM_OPTIONS, (1,), dtype=torch.long).to(device) 

    episode_rewards = deque(maxlen=50)
    episode_lengths = deque(maxlen=50)
    episode_wins = deque(maxlen=50)
    current_episode_reward = 0.0
    current_episode_length = 0

    num_total_updates = MAX_TRAIN_STEPS // NUM_STEPS

    for update in range(start_update, num_total_updates + 1):
        if ANNEAL_LR:
            frac = 1.0 - (update - 1.0) / num_total_updates
            lrnow = frac * LEARNING_RATE
            optimizer.param_groups[0]["lr"] = lrnow

        agent.eval()
        manager_step_idx = 0
        accumulated_manager_reward = 0.0

        for step in range(NUM_STEPS):
            global_step += 1
            current_episode_length += 1

            worker_grids[step] = next_grid
            worker_rules[step] = next_rules
            worker_rule_masks[step] = next_rule_mask
            worker_options[step] = next_manager_option.item() 
            worker_dones[step] = next_worker_done.item()

            is_manager_step = (step % MANAGER_UPDATE_FREQ_K == 0)
            if is_manager_step or next_worker_done.item() > 0.5:
                if step > 0:
                     manager_rewards[manager_step_idx-1] = accumulated_manager_reward
                     manager_dones[manager_step_idx-1] = next_worker_done.item() 

                manager_grids[manager_step_idx] = next_grid
                manager_rules[manager_step_idx] = next_rules
                manager_rule_masks[manager_step_idx] = next_rule_mask

                with torch.no_grad():
                    manager_logits, manager_value = agent.get_manager_option_value(
                        next_grid.unsqueeze(0),
                        next_rules.unsqueeze(0),
                        next_rule_mask.unsqueeze(0),
                    )
                    manager_values[manager_step_idx] = manager_value.flatten()

                manager_probs = Categorical(logits=manager_logits)
                next_manager_option = manager_probs.sample() 
                manager_options_chosen[manager_step_idx] = next_manager_option.item()
                manager_log_probs[manager_step_idx] = manager_probs.log_prob(next_manager_option)
                manager_step_idx += 1
                accumulated_manager_reward = 0.0 


            with torch.no_grad():
                worker_logits, worker_value = agent.get_worker_action_value(
                    next_grid.unsqueeze(0),
                    next_rules.unsqueeze(0),
                    next_rule_mask.unsqueeze(0),
                    next_manager_option 
                )
                worker_values[step] = worker_value.flatten()

            worker_probs = Categorical(logits=worker_logits)
            worker_action = worker_probs.sample() 
            worker_actions[step] = worker_action.item()
            worker_log_probs[step] = worker_probs.log_prob(worker_action)

            obs_dict, reward, done, info = env.step(worker_action.item())
            worker_rewards[step] = torch.tensor(reward).to(device).view(-1)
            accumulated_manager_reward += reward
            current_episode_reward += reward


            next_obs_dict = obs_dict
            next_grid, next_rules, next_rule_mask = dict_to_torch(obs_dict, device)
            next_worker_done = torch.tensor(done, dtype=torch.float32).to(device)

            if done:
                print(
                    f"GS: {global_step}, Upd: {update}/{num_total_updates}, Ep.Reward: {current_episode_reward:.2f}, Ep.Len: {current_episode_length}"
                )
                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_length)
                is_win = 1.0 if reward >= 1.0 else 0.0
                episode_wins.append(is_win)

                writer.add_scalar("charts/episode_reward", current_episode_reward, global_step)
                writer.add_scalar("charts/episode_length", current_episode_length, global_step)
                if len(episode_rewards) > 0:
                    writer.add_scalar("charts/avg_reward_50", np.mean(episode_rewards), global_step)
                    writer.add_scalar("charts/avg_length_50", np.mean(episode_lengths), global_step)
                    writer.add_scalar("charts/win_rate_50", np.mean(episode_wins), global_step)

                current_episode_reward = 0.0
                current_episode_length = 0

                next_obs_dict = env.reset()
                next_grid, next_rules, next_rule_mask = dict_to_torch(next_obs_dict, device)
                next_worker_done = torch.zeros(1).to(device)


        if manager_step_idx > 0: 
             manager_rewards[manager_step_idx-1] = accumulated_manager_reward
             manager_dones[manager_step_idx-1] = next_worker_done.item() 


        agent.eval()
        with torch.no_grad():
            next_worker_value = agent.get_worker_value(
                    next_grid.unsqueeze(0), next_rules.unsqueeze(0), next_rule_mask.unsqueeze(0), next_manager_option
                ).reshape(1, -1)
            worker_advantages = compute_gae(next_worker_value, worker_rewards, worker_dones, worker_values, GAMMA, LAMBDA)
            worker_returns = worker_advantages + worker_values

            num_valid_manager_steps = manager_step_idx
            if num_valid_manager_steps > 0:
                next_manager_value = agent.get_manager_value(
                        next_grid.unsqueeze(0), next_rules.unsqueeze(0), next_rule_mask.unsqueeze(0)
                    ).reshape(1, -1)
                manager_advantages = compute_gae(
                    next_manager_value,
                    manager_rewards[:num_valid_manager_steps],
                    manager_dones[:num_valid_manager_steps],
                    manager_values[:num_valid_manager_steps],
                    GAMMA,
                    LAMBDA
                )
                manager_returns = manager_advantages + manager_values[:num_valid_manager_steps]
            else:
                 manager_advantages = torch.tensor([]).to(device)
                 manager_returns = torch.tensor([]).to(device)


        agent.train()
        b_inds_worker = np.arange(NUM_STEPS)
        b_inds_manager = np.arange(num_valid_manager_steps)

        for epoch in range(EPOCHS):
            np.random.shuffle(b_inds_worker)
            if num_valid_manager_steps > 0:
                np.random.shuffle(b_inds_manager)

            for start in range(0, NUM_STEPS, MINI_BATCH_SIZE):
                end = start + MINI_BATCH_SIZE
                mb_inds = b_inds_worker[start:end]

                mb_grids = worker_grids[mb_inds]
                mb_rules = worker_rules[mb_inds]
                mb_masks = worker_rule_masks[mb_inds]
                mb_options = worker_options[mb_inds] 
                mb_actions = worker_actions[mb_inds] 
                mb_log_probs_old = worker_log_probs[mb_inds]
                mb_advantages = worker_advantages[mb_inds]
                mb_returns = worker_returns[mb_inds]

                new_logits, new_value = agent.get_worker_action_value(mb_grids, mb_rules, mb_masks, mb_options)
                new_probs = Categorical(logits=new_logits)
                new_log_prob = new_probs.log_prob(mb_actions)
                entropy = new_probs.entropy().mean()

                log_ratio = new_log_prob - mb_log_probs_old
                ratio = torch.exp(log_ratio)
                clip_adv = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_advantages
                policy_loss = -torch.min(ratio * mb_advantages, clip_adv).mean()

                new_value = new_value.view(-1)
                value_loss = 0.5 * ((new_value - mb_returns) ** 2).mean()
                loss_worker = policy_loss + VALUE_LOSS_COEF * value_loss - ENTROPY_COEF_WORKER * entropy

                optimizer.zero_grad()
                loss_worker.backward() 
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step() 

            if num_valid_manager_steps > 0:
                 manager_mini_batch_size = min(MINI_BATCH_SIZE, num_valid_manager_steps)
                 if manager_mini_batch_size == 0: continue 

                 for start in range(0, num_valid_manager_steps, manager_mini_batch_size):
                    end = start + manager_mini_batch_size
                    mb_inds = b_inds_manager[start:end]

                    mb_grids = manager_grids[mb_inds]
                    mb_rules = manager_rules[mb_inds]
                    mb_masks = manager_rule_masks[mb_inds]
                    mb_options = manager_options_chosen[mb_inds] 
                    mb_log_probs_old = manager_log_probs[mb_inds]
                    mb_advantages = manager_advantages[mb_inds]
                    mb_returns = manager_returns[mb_inds]

                    new_logits, new_value = agent.get_manager_option_value(mb_grids, mb_rules, mb_masks)
                    new_probs = Categorical(logits=new_logits)
                    new_log_prob = new_probs.log_prob(mb_options)
                    entropy = new_probs.entropy().mean()

                    log_ratio = new_log_prob - mb_log_probs_old
                    ratio = torch.exp(log_ratio)
                    clip_adv = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_advantages.detach()
                    policy_loss = -torch.min(ratio * mb_advantages.detach(), clip_adv).mean()
                    new_value = new_value.view(-1)
                    value_loss = 0.5 * ((new_value - mb_returns) ** 2).mean()
                    loss_manager = policy_loss + VALUE_LOSS_COEF * value_loss - ENTROPY_COEF_MANAGER * entropy

                    optimizer.zero_grad()
                    loss_manager.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                    optimizer.step()


        writer.add_scalar("losses/worker_value_loss", value_loss.item(), global_step) 
        writer.add_scalar("losses/worker_policy_loss", policy_loss.item(), global_step)
        writer.add_scalar("losses/worker_entropy", entropy.item(), global_step)
        if num_valid_manager_steps > 0:
             writer.add_scalar("losses/manager_value_loss", value_loss.item(), global_step) 
             writer.add_scalar("losses/manager_policy_loss", policy_loss.item(), global_step)
             writer.add_scalar("losses/manager_entropy", entropy.item(), global_step)
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        sps = int(NUM_STEPS / (time.time() - start_time))
        writer.add_scalar("charts/SPS", sps, global_step)
        print(f"Update {update}/{num_total_updates}, SPS: {sps}")
        start_time = time.time()


        if update % 50 == 0:  
            checkpoint_path = os.path.join(run_checkpoint_dir, f"hrl_baba_update_{update}.pth")
            torch.save({
                'agent_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step,
                'update': update,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    env.close()
    writer.close()
    print("Training finished.")
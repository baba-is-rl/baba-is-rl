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
from hrl_subgoal_model import SubgoalHierarchicalActorCritic 
from constants import (
    MAX_MAP_HEIGHT, MAX_MAP_WIDTH, MAX_RULES, NUM_OBJECT_TYPES, PAD_INDEX,
    MANAGER_SUBGOAL_FREQ_K, 
    LEVELS, MAPS_DIR, SPRITES_PATH,
    CURRICULUM_THRESHOLD, CURRICULUM_WINDOW,
    CHECKPOINT_DIR_HRL_SUBGOAL, LOG_DIR_HRL_SUBGOAL 
)

LEARNING_RATE = 3e-4
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2
ENTROPY_COEF_WORKER = 0.01 
VALUE_LOSS_COEF = 0.5 
EPOCHS = 10
MINI_BATCH_SIZE = 32 
NUM_STEPS = 1024 
MAX_TRAIN_STEPS = 10_000_000
ANNEAL_LR = True

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
        default=f"hrl_subgoal_curr_{int(time.time())}",
        help="Run name for logs/checkpoints",
    )
    parser.add_argument("--load-checkpoint", "-l", type=str, help="path to checkpoint")
    parser.add_argument("--k", type=int, default=MANAGER_SUBGOAL_FREQ_K, help="Manager subgoal frequency")
    parser.add_argument("--curriculum_threshold", type=float, default=CURRICULUM_THRESHOLD)
    parser.add_argument("--curriculum_window", type=int, default=CURRICULUM_WINDOW)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--num_steps", type=int, default=NUM_STEPS)
    parser.add_argument("--mini_batch_size", type=int, default=MINI_BATCH_SIZE)

    args = parser.parse_args()

    MANAGER_SUBGOAL_FREQ_K = args.k
    CURRICULUM_THRESHOLD = args.curriculum_threshold
    CURRICULUM_WINDOW = args.curriculum_window
    LEARNING_RATE = args.lr
    NUM_STEPS = args.num_steps
    MINI_BATCH_SIZE = args.mini_batch_size

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(args.seed)
        print(f"Using CUDA - K={MANAGER_SUBGOAL_FREQ_K}, State Subgoals, Curriculum")
    else:
        device = torch.device("cpu")
        print(f"Using CPU - K={MANAGER_SUBGOAL_FREQ_K}, State Subgoals, Curriculum")

    os.makedirs(CHECKPOINT_DIR_HRL_SUBGOAL, exist_ok=True)
    os.makedirs(LOG_DIR_HRL_SUBGOAL, exist_ok=True)
    run_log_dir = os.path.join(LOG_DIR_HRL_SUBGOAL, args.run_name)
    run_checkpoint_dir = os.path.join(CHECKPOINT_DIR_HRL_SUBGOAL, args.run_name)
    os.makedirs(run_log_dir, exist_ok=True)
    os.makedirs(run_checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(run_log_dir)
    param_str = "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]))
    param_str += f"\n|NUM_STEPS|{NUM_STEPS}|"
    param_str += f"\n|MINI_BATCH_SIZE|{MINI_BATCH_SIZE}|"
    param_str += f"\n|LEARNING_RATE|{LEARNING_RATE}|"
    writer.add_text("hyperparameters", param_str)

    current_curriculum_stage = 0
    episode_rewards = deque(maxlen=CURRICULUM_WINDOW)
    episode_lengths = deque(maxlen=CURRICULUM_WINDOW)
    episode_wins = deque(maxlen=CURRICULUM_WINDOW)
    global_step = 0
    start_update = 1
    checkpoint_loaded_successfully = False 

    if args.load_checkpoint and os.path.isfile(args.load_checkpoint):
        print(f"Loading checkpoint: {args.load_checkpoint}")
        try:
            checkpoint = torch.load(args.load_checkpoint, map_location=device)
            current_curriculum_stage = checkpoint.get("curriculum_stage", 0)
            episode_wins = deque(checkpoint.get("episode_wins", []), maxlen=CURRICULUM_WINDOW)
            episode_rewards = deque(checkpoint.get("episode_rewards", []), maxlen=CURRICULUM_WINDOW)
            episode_lengths = deque(checkpoint.get("episode_lengths", []), maxlen=CURRICULUM_WINDOW)
            global_step = checkpoint.get("global_step", 0)
            start_update = checkpoint.get("update", (global_step // NUM_STEPS) + 1)
            print(f"-> Resuming at Stage: {current_curriculum_stage}, Global Step: {global_step}, Update: {start_update}")
            checkpoint_loaded_successfully = True 
        except Exception as e:
            print(f"ERROR loading checkpoint state: {e}. Starting fresh.")
            current_curriculum_stage = 0
            episode_wins.clear(); episode_rewards.clear(); episode_lengths.clear()
            global_step = 0; start_update = 1
            checkpoint_loaded_successfully = False
    elif args.load_checkpoint:
        print(f"Warning: Checkpoint file not found: {args.load_checkpoint}. Starting fresh.")
    else:
        print("Starting training from scratch.")

    current_map_name = LEVELS[current_curriculum_stage]
    map_id_part = os.path.splitext(current_map_name.replace("/", "_"))[0]
    env_id = f"baba-{map_id_part}-hrl-subgoal-v0"
    register_env(
        env_id, current_map_name,
        f"{map_id_part} (HRL Subgoal St.{current_curriculum_stage})",
        MAPS_DIR, SPRITES_PATH,
    )
    env = gym.make(env_id)
    env.seed(args.seed)

    grid_shape = env.observation_space["grid"].shape
    num_primitive_actions = env.action_space.n
    agent = SubgoalHierarchicalActorCritic(grid_shape, num_primitive_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)

    if checkpoint_loaded_successfully and args.load_checkpoint:
        try:
            checkpoint = torch.load(args.load_checkpoint, map_location=device) 
            agent.load_state_dict(checkpoint["agent_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("-> Agent and Optimizer weights reloaded successfully.")
        except Exception as e:
             print(f"ERROR loading agent/optimizer state dicts from checkpoint: {e}.")


    worker_grids_u = torch.zeros((NUM_STEPS,) + grid_shape).to(device) 
    worker_rules_u = torch.zeros((NUM_STEPS,) + env.observation_space["rules"].shape, dtype=torch.long).to(device)
    worker_rule_masks_u = torch.zeros((NUM_STEPS,) + env.observation_space["rule_mask"].shape, dtype=torch.long).to(device)
    worker_target_subgoals_w = torch.zeros((NUM_STEPS,) + grid_shape).to(device) 
    worker_actions = torch.zeros(NUM_STEPS, dtype=torch.long).to(device)
    worker_log_probs = torch.zeros(NUM_STEPS).to(device)
    worker_rewards = torch.zeros(NUM_STEPS).to(device)
    worker_dones = torch.zeros(NUM_STEPS).to(device)
    worker_values = torch.zeros(NUM_STEPS).to(device)

    manager_steps_per_update = NUM_STEPS // MANAGER_SUBGOAL_FREQ_K + 2
    manager_grids_u = torch.zeros((manager_steps_per_update,) + grid_shape).to(device) 
    manager_rules_u = torch.zeros((manager_steps_per_update,) + env.observation_space["rules"].shape, dtype=torch.long).to(device)
    manager_rule_masks_u = torch.zeros((manager_steps_per_update,) + env.observation_space["rule_mask"].shape, dtype=torch.long).to(device)
    manager_rewards = torch.zeros(manager_steps_per_update).to(device) 
    manager_dones = torch.zeros(manager_steps_per_update).to(device)   
    manager_values = torch.zeros(manager_steps_per_update).to(device) 

    start_time = time.time()
    next_obs_dict = env.reset()
    next_grid_u, next_rules_u, next_rule_mask_u = dict_to_torch(next_obs_dict, device)
    next_worker_done = torch.zeros(1).to(device)
    with torch.no_grad():
        _, current_target_subgoal_w = agent.get_manager_value_and_subgoal(
            next_grid_u.unsqueeze(0), next_rules_u.unsqueeze(0), next_rule_mask_u.unsqueeze(0)
        )
        current_target_subgoal_w = current_target_subgoal_w.squeeze(0) 

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

            worker_grids_u[step] = next_grid_u
            worker_rules_u[step] = next_rules_u
            worker_rule_masks_u[step] = next_rule_mask_u
            worker_target_subgoals_w[step] = current_target_subgoal_w 
            worker_dones[step] = next_worker_done.item()

            is_manager_step = (step % MANAGER_SUBGOAL_FREQ_K == 0)
            if is_manager_step or next_worker_done.item() > 0.5:
                if step > 0 and manager_step_idx > 0 and manager_step_idx <= len(manager_rewards):
                    manager_rewards[manager_step_idx-1] = accumulated_manager_reward
                    manager_dones[manager_step_idx-1] = next_worker_done.item()

                if manager_step_idx < len(manager_grids_u):
                    manager_grids_u[manager_step_idx] = next_grid_u
                    manager_rules_u[manager_step_idx] = next_rules_u
                    manager_rule_masks_u[manager_step_idx] = next_rule_mask_u

                    with torch.no_grad():
                        manager_value, new_target_subgoal_w = agent.get_manager_value_and_subgoal(
                            next_grid_u.unsqueeze(0),
                            next_rules_u.unsqueeze(0),
                            next_rule_mask_u.unsqueeze(0),
                        )
                        manager_values[manager_step_idx] = manager_value.flatten()
                        current_target_subgoal_w = new_target_subgoal_w.squeeze(0) 


                    manager_step_idx += 1
                    accumulated_manager_reward = 0.0 

            with torch.no_grad():
                w_input = current_target_subgoal_w.unsqueeze(0).to(device)
                worker_logits, worker_value = agent.get_worker_action_value(
                    next_grid_u.unsqueeze(0),
                    next_rules_u.unsqueeze(0),
                    next_rule_mask_u.unsqueeze(0),
                    w_input 
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
            next_grid_u, next_rules_u, next_rule_mask_u = dict_to_torch(obs_dict, device)
            next_worker_done = torch.tensor(done, dtype=torch.float32).to(device)

            if done:
                print(f"GS: {global_step}, Upd: {update}/{num_total_updates}, Stage: {current_curriculum_stage} ({current_map_name}), Ep.R: {current_episode_reward:.2f}, Ep.L: {current_episode_length}")
                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_length)
                is_win = 1.0 if reward >= 1.0 else 0.0
                episode_wins.append(is_win)
                writer.add_scalar("charts/episode_reward", current_episode_reward, global_step)
                writer.add_scalar("charts/episode_length", current_episode_length, global_step)
                if len(episode_wins) > 0: writer.add_scalar("charts/win_rate_window", np.mean(episode_wins), global_step)
                if len(episode_rewards) > 0: writer.add_scalar("charts/avg_reward_window", np.mean(episode_rewards), global_step)
                if len(episode_lengths) > 0: writer.add_scalar("charts/avg_length_window", np.mean(episode_lengths), global_step)
                writer.add_scalar(f"curriculum/stage_{current_curriculum_stage}_reward", current_episode_reward, global_step)
                writer.add_scalar(f"curriculum/stage_{current_curriculum_stage}_win", is_win, global_step)

                new_map_path_for_reset = None
                if len(episode_wins) == CURRICULUM_WINDOW and np.mean(episode_wins) >= CURRICULUM_THRESHOLD:
                    if current_curriculum_stage < len(LEVELS) - 1:
                        current_curriculum_stage += 1
                        current_map_name = LEVELS[current_curriculum_stage]
                        new_map_path_for_reset = os.path.join(MAPS_DIR, current_map_name)
                        print(f"\n--- Advancing to Curriculum Stage {current_curriculum_stage}: {current_map_name} ---")
                        writer.add_text("curriculum/current_level", f"{current_curriculum_stage}: {current_map_name}", global_step)
                        episode_rewards.clear(); episode_lengths.clear(); episode_wins.clear()
                    else:
                        print("\n--- Curriculum Complete! Staying on final level. ---")

                next_obs_dict = env.reset(new_map_path=new_map_path_for_reset)
                next_grid_u, next_rules_u, next_rule_mask_u = dict_to_torch(obs_dict, device)
                next_worker_done = torch.zeros(1).to(device)
                with torch.no_grad():
                     _, current_target_subgoal_w = agent.get_manager_value_and_subgoal(
                         next_grid_u.unsqueeze(0), next_rules_u.unsqueeze(0), next_rule_mask_u.unsqueeze(0)
                     )
                     current_target_subgoal_w = current_target_subgoal_w.squeeze(0)
                current_episode_reward = 0.0
                current_episode_length = 0

        if manager_step_idx > 0 and manager_step_idx <= len(manager_rewards):
             manager_rewards[manager_step_idx-1] = accumulated_manager_reward
             manager_dones[manager_step_idx-1] = next_worker_done.item()

        agent.eval()
        with torch.no_grad():
            next_worker_value = agent.get_worker_value(
                    next_grid_u.unsqueeze(0), next_rules_u.unsqueeze(0), next_rule_mask_u.unsqueeze(0),
                    current_target_subgoal_w.unsqueeze(0).to(device) 
                ).reshape(1, -1)
            worker_advantages = compute_gae(next_worker_value, worker_rewards, worker_dones, worker_values, GAMMA, LAMBDA)
            worker_returns = worker_advantages + worker_values

            num_valid_manager_steps = manager_step_idx
            if num_valid_manager_steps > 0:
                valid_manager_rewards = manager_rewards[:num_valid_manager_steps]
                valid_manager_dones = manager_dones[:num_valid_manager_steps]
                valid_manager_values = manager_values[:num_valid_manager_steps]
                next_manager_value = agent.get_manager_value(
                        next_grid_u.unsqueeze(0), next_rules_u.unsqueeze(0), next_rule_mask_u.unsqueeze(0)
                    ).reshape(1, -1)
                manager_advantages = compute_gae(next_manager_value, valid_manager_rewards, valid_manager_dones, valid_manager_values, GAMMA, LAMBDA)
                manager_returns = manager_advantages + valid_manager_values
            else:
                 manager_advantages = torch.tensor([]).to(device)
                 manager_returns = torch.tensor([]).to(device)

        agent.train()
        b_inds_worker = np.arange(NUM_STEPS)
        b_inds_manager = np.arange(num_valid_manager_steps)
        epoch_worker_policy_loss, epoch_worker_value_loss, epoch_worker_entropy = [], [], []
        epoch_manager_value_loss = [] 
        for epoch in range(EPOCHS):
            np.random.shuffle(b_inds_worker)
            if num_valid_manager_steps > 0: np.random.shuffle(b_inds_manager)

            for start in range(0, NUM_STEPS, MINI_BATCH_SIZE):
                end = start + MINI_BATCH_SIZE
                mb_inds = b_inds_worker[start:end]

                mb_grids_u = worker_grids_u[mb_inds]
                mb_rules_u = worker_rules_u[mb_inds]
                mb_masks_u = worker_rule_masks_u[mb_inds]
                mb_target_w = worker_target_subgoals_w[mb_inds] 
                mb_actions = worker_actions[mb_inds]
                mb_log_probs_old = worker_log_probs[mb_inds]
                mb_advantages = worker_advantages[mb_inds]
                mb_returns = worker_returns[mb_inds]

                new_logits, new_value = agent.get_worker_action_value(mb_grids_u, mb_rules_u, mb_masks_u, mb_target_w)
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

                epoch_worker_policy_loss.append(policy_loss.item())
                epoch_worker_value_loss.append(value_loss.item())
                epoch_worker_entropy.append(entropy.item())

            if num_valid_manager_steps > 0:
                 manager_mini_batch_size = min(MINI_BATCH_SIZE, num_valid_manager_steps)
                 if manager_mini_batch_size == 0: continue

                 for start in range(0, num_valid_manager_steps, manager_mini_batch_size):
                    end = start + manager_mini_batch_size
                    mb_inds = b_inds_manager[start:end]

                    mb_grids_u = manager_grids_u[mb_inds]
                    mb_rules_u = manager_rules_u[mb_inds]
                    mb_masks_u = manager_rule_masks_u[mb_inds]
                    mb_returns = manager_returns[mb_inds] 

                    new_value = agent.get_manager_value(mb_grids_u, mb_rules_u, mb_masks_u)

                    new_value = new_value.view(-1)
                    value_loss = 0.5 * ((new_value - mb_returns) ** 2).mean()
                    loss_manager = VALUE_LOSS_COEF * value_loss 

                    optimizer.zero_grad()
                    loss_manager.backward() 
                    nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                    optimizer.step()

                    epoch_manager_value_loss.append(value_loss.item())


        writer.add_scalar("losses/worker_policy_loss", np.mean(epoch_worker_policy_loss), global_step)
        writer.add_scalar("losses/worker_value_loss", np.mean(epoch_worker_value_loss), global_step)
        writer.add_scalar("losses/worker_entropy", np.mean(epoch_worker_entropy), global_step)
        if len(epoch_manager_value_loss) > 0:
            writer.add_scalar("losses/manager_value_loss", np.mean(epoch_manager_value_loss), global_step)
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        sps = int(NUM_STEPS / (time.time() - start_time))
        writer.add_scalar("charts/SPS", sps, global_step)
        print(f"Update {update}/{num_total_updates}, SPS: {sps}")
        start_time = time.time()

        if update % 20 == 0:
            checkpoint_path = os.path.join(run_checkpoint_dir, f"hrl_subgoal_update_{update}.pth")
            torch.save({
                'agent_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step, 'update': update,
                'curriculum_stage': current_curriculum_stage,
                'episode_wins': list(episode_wins), 'episode_rewards': list(episode_rewards),
                'episode_lengths': list(episode_lengths),
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    env.close()
    writer.close()
    print("Training finished.")
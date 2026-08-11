"""PPO trainer for collective communication optimization."""

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .buffer import SlotBuffer
from ..envs.decoder import SlotDecoder, recompute_logp_slot
from ..envs.evaluator import evaluate_schedule
from ..envs.problem import compute_received_chunks


def compute_gae_advantages(rewards, values, dones, gamma, gae_lambda):
    """Compute Generalized Advantage Estimation.
    
    Args:
        rewards: Tensor of rewards, shape (T,)
        values: Tensor of value estimates, shape (T,)
        dones: Tensor of done flags, shape (T,)
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        
    Returns:
        advantages: GAE advantages, shape (T,)
        returns: Returns (advantages + values), shape (T,)
    """
    advantages = []
    gae = 0.0
    
    for t in reversed(range(len(rewards))):
        if dones[t]:
            gae = 0.0
            next_val = 0.0
        else:
            next_val = values[t + 1].item() if t + 1 < len(values) else 0.0
        
        delta = rewards[t] + gamma * next_val - values[t]
        gae = delta + gamma * gae_lambda * gae
        advantages.insert(0, gae)
    
    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = advantages + values
    
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return advantages, returns


def train_epoch(model, optimizer, train_problems, device, config, epoch, total_epochs):
    """Train for one epoch.
    
    Args:
        model: SlotLevelPolicy model
        optimizer: Persistent optimizer owned by the training script
        train_problems: List of (scenario_id, ProblemInstance)
        device: torch device
        config: Training configuration dict
        epoch: Current epoch number (0-indexed)
        total_epochs: Total number of epochs
        
    Returns:
        avg_policy_loss: Average policy loss
        avg_value_loss: Average value loss
        avg_entropy: Average entropy
        total_reward: Total average reward
    """
    model.train()
    slot_buffer = SlotBuffer()
    
    # Collect experience
    indices = np.random.permutation(len(train_problems))
    slots_collected = 0
    batch_target = config['batch_target']
    
    pbar = tqdm(indices, desc=f"Epoch {epoch+1}/{total_epochs} - Collecting")
    
    for idx in pbar:
        if slots_collected >= batch_target:
            break
        
        scenario_id, problem = train_problems[idx]
        if np.sum(problem.demands) == 0:
            continue
        
        topo_info = getattr(problem, 'topology_info', None)
        if topo_info is None:
            continue
        
        decoder = SlotDecoder(topo_info)
        state = problem.initial_state.copy()
        demands = problem.demands.copy()
        initial_total_demands = max(1.0, float(np.sum(problem.demands)))
        
        # Get static info once per problem (shared across all slots of this problem)
        static_info = decoder.get_static_info()
        
        for t in range(problem.T):
            Y_t, logp_slot, entropy_slot, value, state_info, micro_actions = decoder.decode_slot(
                model,
                state,
                demands,
                t,
                problem.T,
                train=True,
                moment_context=(
                    getattr(problem, 'moment_context', None)
                    if getattr(model, 'global_moment_feat_dim', 0) > 0
                    else None
                ),
                current_matrix=getattr(problem, 'traffic_matrix', None),
                moment_max_entry=config.get('moment_max_entry', 8.0),
            )
            
            N_t = compute_received_chunks(Y_t, topo_info.edge_dst, topo_info.V)
            state = np.maximum(state, N_t)
            demands = demands * (1 - N_t)
            
            remaining = float(np.sum(demands))
            slot_reward = -remaining / initial_total_demands
            
            episode_success = (np.sum(demands) == 0)
            episode_timeout = (t == problem.T - 1)
            episode_end = episode_success or episode_timeout
            
            slot_buffer.add(
                state_info=state_info,
                actions=micro_actions,
                logprob_slot=logp_slot.detach().cpu(),
                value=value.detach().cpu(),
                reward=slot_reward,
                done=episode_end,
                static_info=static_info  # Shared reference, not copied
            )
            
            slots_collected += 1
            pbar.set_postfix({'slots': slots_collected, 'target': batch_target})
            
            if episode_end:
                break
    
    if len(slot_buffer) == 0:
        print("  No slots collected, skipping update")
        return 0.0, 0.0, 0.0, 0.0
    
    # Compute advantages
    rewards = torch.tensor(slot_buffer.slot_rewards, dtype=torch.float32)
    values = torch.tensor([v.item() for v in slot_buffer.slot_values], dtype=torch.float32)
    dones = torch.tensor(slot_buffer.slot_dones, dtype=torch.float32)
    
    advantages, returns = compute_gae_advantages(
        rewards, values, dones, config['gamma'], config['gae_lambda']
    )
    
    # PPO Update
    buffer_size = len(slot_buffer)
    indices_ppo = np.arange(buffer_size)
    
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_reward = rewards.mean().item()
    num_updates = 0
    
    for ppo_epoch in range(config['ppo_epochs']):
        np.random.shuffle(indices_ppo)
        mb_size = config['mini_batch_size']
        
        pbar_ppo = tqdm(range(0, buffer_size, mb_size), 
                       desc=f"  PPO Epoch {ppo_epoch+1}/{config['ppo_epochs']}")
        
        for start in pbar_ppo:
            end = min(start + mb_size, buffer_size)
            batch_indices = indices_ppo[start:end]
            
            optimizer.zero_grad()
            loss_accum = 0.0
            policy_loss_accum = 0.0
            value_loss_accum = 0.0
            entropy_accum = 0.0
            
            for idx_sample in batch_indices:
                state_info = slot_buffer.slot_states[idx_sample]
                micro_actions = slot_buffer.slot_actions[idx_sample]
                static_info = slot_buffer.slot_static_infos[idx_sample]
                old_logp = slot_buffer.slot_logprobs[idx_sample].to(device)
                advantage = advantages[idx_sample].to(device)
                ret = returns[idx_sample].to(device)
                
                logp_new, entropy_new, value_new = recompute_logp_slot(
                    model, state_info, micro_actions, device, static_info
                )
                
                ratio = torch.exp(logp_new - old_logp)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - config['clip_eps'], 
                                   1.0 + config['clip_eps']) * advantage
                policy_loss = -torch.min(surr1, surr2)
                
                value_loss = F.mse_loss(value_new.squeeze(), ret)
                loss = (policy_loss + config['value_coef'] * value_loss 
                       - config['entropy_coef'] * entropy_new)
                loss_accum += loss
                
                policy_loss_accum += policy_loss.item()
                value_loss_accum += value_loss.item()
                entropy_accum += entropy_new.item()
            
            loss_accum = loss_accum / len(batch_indices)
            loss_accum.backward()
            
            if 'max_grad_norm' in config:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            
            optimizer.step()
            
            total_policy_loss += policy_loss_accum
            total_value_loss += value_loss_accum
            total_entropy += entropy_accum
            num_updates += len(batch_indices)
            
            pbar_ppo.set_postfix({
                'p_loss': policy_loss_accum / len(batch_indices),
                'v_loss': value_loss_accum / len(batch_indices),
            })
    
    avg_policy_loss = total_policy_loss / max(num_updates, 1)
    avg_value_loss = total_value_loss / max(num_updates, 1)
    avg_entropy = total_entropy / max(num_updates, 1)
    
    return avg_policy_loss, avg_value_loss, avg_entropy, total_reward


def evaluate_model(model, test_problems, device, moment_max_entry=8.0):
    """Evaluate model on test set.
    
    Args:
        model: SlotLevelPolicy model
        test_problems: List of (scenario_id, ProblemInstance)
        device: torch device
        
    Returns:
        avg_score: Average evaluation score
        avg_steps: Average completion steps
    """
    model.eval()
    scores = []
    completion_steps = []
    
    with torch.no_grad():
        for scenario_id, problem in tqdm(test_problems, desc="Evaluating"):
            topo_info = getattr(problem, 'topology_info', None)
            if topo_info is None:
                continue
            
            decoder = SlotDecoder(topo_info)
            state = problem.initial_state.copy()
            demands = problem.demands.copy()
            schedule = []
            
            for t in range(problem.T):
                Y_t, _, _, _, _, _ = decoder.decode_slot(
                    model,
                    state,
                    demands,
                    t,
                    problem.T,
                    train=False,
                    moment_context=(
                        getattr(problem, 'moment_context', None)
                        if getattr(model, 'global_moment_feat_dim', 0) > 0
                        else None
                    ),
                    current_matrix=getattr(problem, 'traffic_matrix', None),
                    moment_max_entry=moment_max_entry,
                )
                schedule.append(Y_t)
                
                N_t = compute_received_chunks(Y_t, topo_info.edge_dst, topo_info.V)
                state = np.maximum(state, N_t)
                demands = demands * (1 - N_t)
                
                if np.sum(demands) == 0:
                    completion_steps.append(t + 1)
                    break
            else:
                completion_steps.append(problem.T)
            
            while len(schedule) < problem.T:
                schedule.append(np.zeros((problem.C, problem.E), dtype=int))
            
            score, error = evaluate_schedule(schedule, problem)
            if error == "":  # Empty string means success, not None
                scores.append(score)
            else:
                # Log the error for debugging
                print(f"  Warning: Problem {scenario_id} evaluation failed: {error}")
    
    if scores:
        avg_score = np.mean(scores)
        avg_steps = np.mean(completion_steps)
    else:
        # Return a very poor score instead of 0.0, since score is negative (closer to 0 is better)
        # Use -(T_max + 10) as default poor score
        avg_score = -1000.0
        avg_steps = 0.0
    
    return avg_score, avg_steps

#!/usr/bin/env python3
"""Training script for RLCCL - Simple and clean interface."""

import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from rlccl.config import get_config
from rlccl.envs.problem import ProblemInstance
from rlccl.envs.evaluator import (
    LEGACY_GENERATOR_NAME,
    generate_scenarios, 
    load_or_generate_scenarios as load_or_generate_scenario_dicts,
    load_topology_info,
    build_problem_from_scenario,
)
from rlccl.models import SlotLevelPolicy
from rlccl.training import train_epoch, evaluate_model


def load_or_generate_scenarios(topology, num_scenarios, seed, is_train=True):
    """Load scenarios from JSON or generate if not exists."""
    data_dir = Path('Data') / topology
    data_dir.mkdir(parents=True, exist_ok=True)
    
    prefix = 'train' if is_train else 'test'
    json_file = data_dir / (
        f"scenarios_{LEGACY_GENERATOR_NAME}_{prefix}_{num_scenarios}_seed{seed}.json"
    )
    topology_info = load_topology_info(topology)
    existed = json_file.exists()
    scenarios = load_or_generate_scenario_dicts(
        str(json_file), topology_info.V, num_scenarios, seed
    )
    problems = []
    for scenario in scenarios:
        problem = build_problem_from_scenario(
            V=topology_info.V,
            E=topology_info.E,
            edges=topology_info.edges,
            capacities=topology_info.capacities,
            shared_constraints=topology_info.shared_constraints,
            scenario=scenario,
            T=20
        )
        scenario_id = scenario.get('id', len(problems))
        problems.append((f"{topology}_{prefix}_{scenario_id}", problem))
    action = "Loaded" if existed else "Generated"
    print(f"    OK: {action} {len(problems)} scenarios via {json_file.name}")
    return problems


def load_topology_data(topology, num_train, num_test, seed):
    """Load or generate training and testing data for a topology."""
    print(f"  {topology}:")
    train_data = load_or_generate_scenarios(topology, num_train, seed, is_train=True)
    test_data = load_or_generate_scenarios(topology, num_test, seed + 1000, is_train=False)
    print(f"    Total: Train={len(train_data)}, Test={len(test_data)}")
    return train_data, test_data


def main():
    parser = argparse.ArgumentParser(description='Train RLCCL model - Simple interface')
    
    # Data parameters - SIMPLE!
    parser.add_argument('--topology', nargs='+', default=['Rear8GPU_NoSwitch_Test'],
                       help='Topology names (space-separated)')
    parser.add_argument('--num_train', type=int, default=80,
                       help='Training scenarios per topology')
    parser.add_argument('--num_test', type=int, default=30,
                       help='Test scenarios per topology')
    parser.add_argument('--seed', type=int, default=42)
    
    # Training parameters
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_target', type=int, default=500)
    parser.add_argument('--ppo_epochs', type=int, default=10)
    parser.add_argument('--mini_batch_size', type=int, default=32)
    parser.add_argument('--eval_interval', type=int, default=5)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume', type=str, default=None)
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = get_config()
    config.update({
        'hidden_dim': args.hidden_dim,
        'lr': args.lr,
        'batch_target': args.batch_target,
        'ppo_epochs': args.ppo_epochs,
        'mini_batch_size': args.mini_batch_size,
    })
    
    # Load data
    print(f"\n{'='*80}")
    print(f"Loading Data")
    print(f"{'='*80}")
    print(f"Topologies: {', '.join(args.topology)}")
    print(f"Per topology: {args.num_train} train, {args.num_test} test")
    print(f"Seed: {args.seed}\n")
    
    all_train_data = []
    all_test_data = []
    
    for topo in args.topology:
        train_data, test_data = load_topology_data(topo, args.num_train, args.num_test, args.seed)
        all_train_data.extend([(f"{topo}_{sid}", prob) for sid, prob in train_data])
        all_test_data.extend([(f"{topo}_{sid}", prob) for sid, prob in test_data])
    
    if not all_train_data:
        print("ERROR: No training data!")
        return
    
    print(f"\n{'='*80}")
    print(f"Summary: {len(args.topology)} topologies, {len(all_train_data)} train, {len(all_test_data)} test")
    print(f"{'='*80}\n")
    
    model_name = args.topology[0] if len(args.topology) == 1 else "multi_topology"
    
    _, example_problem = all_train_data[0]
    
    device = torch.device(args.device)
    model = SlotLevelPolicy(
        node_feat_dim=5, edge_feat_dim=2,
        cand_feat_dim=5, chunk_feat_dim=2,
        hidden_dim=config['hidden_dim'],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    
    start_epoch = 0
    best_score = -float('inf')
    
    if args.resume:
        print(f"Resuming from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_score = checkpoint.get('best_score', -float('inf'))
            print(f"  Epoch {start_epoch}, best={best_score:.4f}\n")
        else:
            model.load_state_dict(checkpoint)
    
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters\n")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print(f"{'='*80}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*80}")
        
        policy_loss, value_loss, entropy, avg_reward = train_epoch(
            model, optimizer, all_train_data, device, config, epoch, args.epochs
        )
        
        print(f"\nMetrics: policy={policy_loss:.4f}, value={value_loss:.4f}, "
              f"entropy={entropy:.4f}, reward={avg_reward:.4f}")
        
        if (epoch + 1) % args.eval_interval == 0 and all_test_data:
            print(f"\nEvaluating...")
            avg_score, avg_steps = evaluate_model(model, all_test_data, device)
            print(f"  Score={avg_score:.4f}, Steps={avg_steps:.2f}")
            
            if avg_score > best_score:
                best_score = avg_score
                best_path = output_dir / f"{model_name}_best.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_score': best_score,
                    'config': config,
                    'topologies': args.topology,
                }, best_path)
                print("  OK: Best model saved")
        
        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = output_dir / f"{model_name}_epoch{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_score': best_score,
                'config': config,
                'topologies': args.topology,
            }, ckpt_path)
            print("  OK: Checkpoint saved")
    
    final_path = output_dir / f"{model_name}_final.pth"
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_score': best_score,
        'config': config,
        'topologies': args.topology,
    }, final_path)
    print(f"\nDone! Best score: {best_score:.4f}")


if __name__ == '__main__':
    main()

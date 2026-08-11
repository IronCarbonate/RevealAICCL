#!/usr/bin/env python3
"""Quick start example for Evolved CCL."""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evolved_ccl import SlotLevelPolicy, get_config
from evolved_ccl.envs import load_topology_info, generate_train_test_split

def main():
    print("=" * 80)
    print("Evolved CCL - Quick Start Example")
    print("=" * 80)
    
    # Load configuration
    config = get_config()
    print(f"\nDefault configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create model
    print(f"\nCreating model with hidden_dim={config['hidden_dim']}")
    model = SlotLevelPolicy(
        node_feat_dim=5,
        edge_feat_dim=2,
        cand_feat_dim=5,
        chunk_feat_dim=2,
        hidden_dim=config['hidden_dim']
    )
    
    print(f"Model created successfully!")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Load topology
    print(f"\nLoading topology: Rear8GPU_NoSwitch_Test")
    topology_info = load_topology_info('Rear8GPU_NoSwitch_Test')
    print(f"  Nodes: {topology_info.V}")
    print(f"  Edges: {topology_info.E}")
    print(f"  Shared constraints: {len(topology_info.shared_constraints)}")
    
    # Generate data
    print(f"\nGenerating training/test data...")
    train_scenarios, test_scenarios = generate_train_test_split(
        topologies=['Rear8GPU_NoSwitch_Test'],
        num_train_per_topo=10,
        num_test_per_topo=5
    )
    print(f"  Training scenarios: {len(train_scenarios)}")
    print(f"  Test scenarios: {len(test_scenarios)}")
    
    print(f"\n✓ Setup complete! Ready to train.")
    print(f"\nNext steps:")
    print(f"  1. Run: python scripts/train.py --epochs 5")
    print(f"  2. Run: python scripts/test.py --model_path checkpoints/final_model.pth")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Testing script for RLCCL - Simple and clean interface."""

import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from rlccl.envs.decoder import SlotDecoder
from rlccl.envs.evaluator import (
    LEGACY_GENERATOR_NAME,
    evaluate_schedule, 
    generate_scenarios,
    load_or_generate_scenarios as load_or_generate_scenario_dicts,
    load_topology_info,
    build_problem_from_scenario,
)
from rlccl.envs.problem import ProblemInstance, compute_received_chunks
from rlccl.models import SlotLevelPolicy


def schedule_to_torch_tensors(schedule, topo_info):
    """Convert schedule to torch tensors for XML export."""
    schedule_array = np.stack(schedule, axis=0)
    real_strategy_list = torch.from_numpy(schedule_array).unsqueeze(0).float()
    
    edge_src_idx = torch.from_numpy(topo_info.edge_src).unsqueeze(0).long()
    edge_dst_idx = torch.from_numpy(topo_info.edge_dst).unsqueeze(0).long()
    
    C, V = len(topo_info.chunk_nodes), len(topo_info.is_switch)
    pre_condition = np.zeros((C, V), dtype=int)
    for c, v in enumerate(topo_info.chunk_nodes):
        pre_condition[c, v] = 1
    pre_condition_tensor = torch.from_numpy(pre_condition).unsqueeze(0).long()
    
    is_switch = torch.from_numpy(topo_info.is_switch.astype(np.int64)).long()
    chunk_mask = torch.ones(1, C, dtype=torch.bool)
    
    return {
        'real_strategy_list': real_strategy_list,
        'edge_src_idx': edge_src_idx,
        'edge_dst_idx': edge_dst_idx,
        'pre_condition': pre_condition_tensor,
        'is_switch': is_switch,
        'chunk_mask': chunk_mask,
    }


def load_or_generate_scenarios(topology, num_scenarios, seed, is_train=False):
    """Load scenarios from JSON or generate if not exists."""
    data_dir = Path('Data') / topology
    data_dir.mkdir(parents=True, exist_ok=True)
    
    prefix = 'train' if is_train else 'test'
    json_file = data_dir / (
        f"scenarios_{LEGACY_GENERATOR_NAME}_{prefix}_{num_scenarios}_seed{seed}.json"
    )
    topology_info = load_topology_info(topology)
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
    
    return problems


def test_model(model, problem, device, export_schedule=False):
    """Run model inference on a problem instance."""
    model.eval()
    topo_info = getattr(problem, 'topology_info', None)
    if topo_info is None:
        return 0.0, problem.T, None
    
    decoder = SlotDecoder(topo_info)
    state = problem.initial_state.copy()
    demands = problem.demands.copy()
    schedule = []
    completion_step = problem.T
    
    with torch.no_grad():
        for t in range(problem.T):
            Y_t, _, _, _, _, _ = decoder.decode_slot(
                model, state, demands, t, problem.T, train=False
            )
            schedule.append(Y_t)
            
            N_t = compute_received_chunks(Y_t, topo_info.edge_dst, topo_info.V)
            state = np.maximum(state, N_t)
            demands = demands * (1 - N_t)
            
            if np.sum(demands) == 0:
                completion_step = t + 1
                break
    
    while len(schedule) < problem.T:
        schedule.append(np.zeros((problem.C, problem.E), dtype=int))
    
    score, error = evaluate_schedule(schedule, problem)
    
    if error:
        print(f"  Warning: {error}")
        score = 0.0
    
    if export_schedule:
        return score, completion_step, schedule
    else:
        return score, completion_step, None


def main():
    parser = argparse.ArgumentParser(description='Test RLCCL model - Simple interface')
    
    # Model and data
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--topology', nargs='+', default=['Rear8GPU_NoSwitch_Test'],
                       help='Topology names (space-separated)')
    parser.add_argument('--num_test', type=int, default=30,
                       help='Test scenarios per topology')
    parser.add_argument('--seed', type=int, default=42)
    
    # Output
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--export_xml', action='store_true',
                       help='Export MSCCL XML files')
    
    # Model config
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    device = torch.device(args.device)
    print(f"Loading model from {args.model_path}...")
    
    model = SlotLevelPolicy(
        node_feat_dim=5, edge_feat_dim=2,
        cand_feat_dim=5, chunk_feat_dim=2,
        hidden_dim=args.hidden_dim,
    ).to(device)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Epoch {checkpoint.get('epoch', '?')}")
    else:
        model.load_state_dict(checkpoint)
    
    # Load test data
    print(f"\nLoading test data...")
    print(f"Topologies: {', '.join(args.topology)}")
    print(f"Per topology: {args.num_test} test scenarios\n")
    
    all_test_data = []
    for topo in args.topology:
        print(f"  {topo}:")
        test_data = load_or_generate_scenarios(topo, args.num_test, args.seed + 1000, is_train=False)
        all_test_data.extend([(f"{topo}_{sid}", prob) for sid, prob in test_data])
        print(f"    OK: {len(test_data)} scenarios")
    
    if not all_test_data:
        print("ERROR: No test data!")
        return
    
    print(f"\nTotal: {len(all_test_data)} scenarios\n")
    
    # Test each scenario
    scores = []
    completion_steps = []
    
    print(f"{'='*80}")
    print("Testing")
    print(f"{'='*80}")
    
    for scenario_id, problem in all_test_data:
        score, steps, schedule = test_model(model, problem, device, export_schedule=args.export_xml)
        
        scores.append(score)
        completion_steps.append(steps)
        
        print(f"{scenario_id}: score={score:.4f}, steps={steps}")
        
        # Export human-readable strategy
        strategy_file = output_dir / f"{scenario_id}_strategy.txt"
        with open(strategy_file, 'w') as f:
            f.write(f"Scenario: {scenario_id}\n")
            f.write(f"Score: {score:.4f}\n")
            f.write(f"Steps: {steps}/{problem.T}\n")
            f.write(f"Topology: V={problem.V}, E={problem.E}, C={problem.C}\n\n")
            
            if schedule:
                for t, Y_t in enumerate(schedule):
                    f.write(f"=== Step {t} ===\n")
                    active_edges = np.where(Y_t.sum(axis=0) > 0)[0]
                    if len(active_edges) == 0:
                        f.write("  (no active edges)\n")
                    else:
                        for e in active_edges:
                            chunks = np.where(Y_t[:, e] > 0)[0]
                            f.write(f"  Edge {e}: chunks {list(chunks)}\n")
                    f.write("\n")
        
        # Export XML if requested
        if args.export_xml and schedule:
            try:
                from rlccl.utils import tensors_to_msccl_xml

                tensors = schedule_to_torch_tensors(schedule, problem.topology_info)
                xml_str = tensors_to_msccl_xml(
                    tensors=tensors,
                    collective="allgather",
                    protocol="Simple",
                    inplace=False,
                )
                
                xml_file = output_dir / f"{scenario_id}.xml"
                with open(xml_file, 'w') as f:
                    f.write(xml_str)
            except Exception as e:
                print(f"  Warning: XML export failed: {e}")
    
    # Summary
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    print(f"Total scenarios: {len(all_test_data)}")
    print(f"Avg score:       {np.mean(scores):.4f}")
    print(f"Avg steps:       {np.mean(completion_steps):.2f}")
    print(f"Min/Max score:   {np.min(scores):.4f} / {np.max(scores):.4f}")
    print(f"Min/Max steps:   {np.min(completion_steps)} / {np.max(completion_steps)}")


if __name__ == '__main__':
    main()

"""
Evaluator for Collective Communication Optimization

This evaluator supports a TRAIN-THEN-GENERALIZE paradigm:
1. Training Phase: Train a neural network on a set of training scenarios
2. Evaluation Phase: Test the trained model on unseen test scenarios

The evolved program should implement:
- A neural network model that can generalize across instances
- A training function that learns from multiple scenarios
- An inference function that applies the trained model to new instances
"""

import numpy as np
import tempfile
import os
import sys
import traceback
import importlib.util
import json
import subprocess
import pickle
import hashlib

try:
    from .problem import ProblemInstance, TopologyInfo, compute_received_chunks
    from ..traffic.matrix_utils import traffic_matrix_to_scenario
except ImportError:  # Support the evaluator's legacy direct-file execution path.
    from rlccl.envs.problem import ProblemInstance, TopologyInfo, compute_received_chunks
    from rlccl.traffic.matrix_utils import traffic_matrix_to_scenario


SCENARIO_SCHEMA_VERSION = 2
LEGACY_GENERATOR_NAME = "legacy_mixed_v2"

def load_topology_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    nodes = data.get('nodes', [])
    links = data.get('links', [])
    
    # Map node id to index 0..V-1
    node_id_map = {n['id']: i for i, n in enumerate(nodes)}
    V = len(nodes)
    
    edges = []
    capacities = []
    edge_map = {} # (u, v) -> edge_index
    
    for i, link in enumerate(links):
        u = node_id_map[link['source']]
        v = node_id_map[link['target']]
        cap = link.get('capacity_value', 1.0)
        
        edges.append([u, v])
        capacities.append(cap)
        edge_map[(u, v)] = i
        
    edges = np.array(edges)
    capacities = np.array(capacities)
    E = len(edges)
    
    # Parse shared constraints from bandwidth_groups
    shared_constraints = []
    bandwidth_groups = data.get('bandwidth_groups', {})
    
    for group_name, group_data in bandwidth_groups.items():
        if isinstance(group_data, dict) and 'edges' in group_data and 'max_bandwidth' in group_data:
            limit = group_data['max_bandwidth']
            constrained_edges = []
            for edge_def in group_data['edges']:
                u = node_id_map[edge_def['source']]
                v = node_id_map[edge_def['target']]
                if (u, v) in edge_map:
                    constrained_edges.append(edge_map[(u, v)])
            
            if constrained_edges:
                shared_constraints.append((constrained_edges, limit))
                
    return V, E, edges, capacities, shared_constraints

def generate_all_to_all_v_scenario(V, rng):
    """
    Generate an All-to-All-V scenario.
    
    All-to-All-V: V×V traffic matrix where:
    - Diagonal is 0 (no self-send)
    - Off-diagonal entries are random in [0, 4]
    
    Each entry traffic_matrix[i][j] = k means node i needs to send k chunks to node j.
    We model this as: node i has chunks destined for node j.
    
    Args:
        V: Number of nodes
        rng: numpy random generator
        
    Returns:
        dict with 'type', 'traffic_matrix', and derived 'initial_state', 'demands'
    """
    # Generate V×V traffic matrix.  The old code used integers(1, 5), which
    # contradicted the documented inclusive [0, 4] range and never made zeros.
    traffic_matrix = rng.integers(0, 5, size=(V, V))
    np.fill_diagonal(traffic_matrix, 0)  # No self-send
    if not np.any(traffic_matrix):
        if V < 2:
            raise ValueError("All-to-All-V requires at least two nodes")
        traffic_matrix[0, 1] = 1
    return traffic_matrix_to_scenario(traffic_matrix)


def generate_allgather_scenario(V, chunks_per_node, rng):
    """
    Generate an AllGather scenario.
    
    AllGather: Each node i initially has `chunks_per_node` chunks.
    After AllGather, every node should have all chunks from all nodes.
    
    Args:
        V: Number of nodes
        chunks_per_node: Number of chunks each node initially has
        rng: numpy random generator
        
    Returns:
        dict with scenario info
    """
    chunks_per_node = int(chunks_per_node)
    C = V * chunks_per_node  # Total chunks
    
    initial_state = np.zeros((C, V), dtype=int)
    demands = np.zeros((C, V), dtype=int)
    
    for node in range(V):
        for k in range(chunks_per_node):
            chunk_idx = node * chunks_per_node + k
            initial_state[chunk_idx, node] = 1  # Node owns this chunk
            
            # All OTHER nodes need this chunk
            for other in range(V):
                if other != node:
                    demands[chunk_idx, other] = 1
    
    return {
        'type': 'allgather',
        'chunks_per_node': int(chunks_per_node),
        'C': int(C),
        'initial_state': [[int(x) for x in row] for row in initial_state],
        'demands': [[int(x) for x in row] for row in demands]
    }


def generate_alltoall_scenario(V, chunks_per_dest, rng):
    """
    Generate an AllToAll scenario.
    
    AllToAll (transpose): 
    - Think of a V×V matrix of data blocks, where block[i][j] needs to go from node i to node j.
    - Initially, node i has all blocks in row i: block[i][0], block[i][1], ..., block[i][V-1]
    - After AllToAll, node j has all blocks in column j: block[0][j], block[1][j], ..., block[V-1][j]
    
    For blocks where i==j, they are already at the correct node (no transfer needed).
    
    Args:
        V: Number of nodes
        chunks_per_dest: Number of chunks per (src, dst) pair
        rng: numpy random generator
        
    Returns:
        dict with scenario info
    """
    chunks_per_dest = int(chunks_per_dest)
    
    # Total chunks = V * V * chunks_per_dest
    # But we only need to transfer chunks where src != dst
    # For simplicity, we model ALL chunks, but demands[chunk][dst] = 0 when src == dst
    
    C = V * V * chunks_per_dest
    
    initial_state = np.zeros((C, V), dtype=int)
    demands = np.zeros((C, V), dtype=int)
    
    # Chunk indexing: chunk[src][dst][k] = src * V * chunks_per_dest + dst * chunks_per_dest + k
    # This chunk is initially at node src, and needs to go to node dst
    for src in range(V):
        for dst in range(V):
            for k in range(chunks_per_dest):
                chunk_idx = src * V * chunks_per_dest + dst * chunks_per_dest + k
                initial_state[chunk_idx, src] = 1  # src has this chunk
                
                # Only dst needs this chunk if src != dst
                # If src == dst, the chunk is already at the right place
                if src != dst:
                    demands[chunk_idx, dst] = 1
    
    return {
        'type': 'alltoall',
        'chunks_per_dest': int(chunks_per_dest),
        'C': int(C),
        'initial_state': [[int(x) for x in row] for row in initial_state],
        'demands': [[int(x) for x in row] for row in demands]
    }


def generate_scenarios(V, num_scenarios, seed):
    """
    Generate a diverse set of collective communication scenarios.
    
    Mix of:
    - All-to-All-V (random traffic matrix)
    - AllGather (with varying chunks_per_node)
    - AllToAll (transpose pattern)
    
    Args:
        V: Number of nodes
        num_scenarios: Total number of scenarios
        seed: Random seed
        
    Returns:
        List of scenario dicts
    """
    rng = np.random.default_rng(seed)
    scenarios = []
    
    # Distribution: 50% All-to-All-V, 30% AllGather, 20% AllToAll
    num_a2av = int(num_scenarios * 0.5)
    num_allgather = int(num_scenarios * 0.3)
    num_alltoall = num_scenarios - num_a2av - num_allgather
    
    # Generate All-to-All-V scenarios
    for i in range(num_a2av):
        scenario = generate_all_to_all_v_scenario(V, rng)
        scenario['id'] = len(scenarios)
        scenarios.append(scenario)
    
    # Generate AllGather scenarios with varying chunks_per_node
    for i in range(num_allgather):
        chunks_per_node = rng.integers(1, 4)  # 1, 2, or 3 chunks per node
        scenario = generate_allgather_scenario(V, chunks_per_node, rng)
        scenario['id'] = len(scenarios)
        scenarios.append(scenario)
    
    # Generate AllToAll scenarios
    for i in range(num_alltoall):
        chunks_per_dest = 1  # 1 or 2 chunks per (src, dst) pair
        scenario = generate_alltoall_scenario(V, chunks_per_dest, rng)
        scenario['id'] = len(scenarios)
        scenarios.append(scenario)
    
    # Shuffle to mix scenario types
    rng.shuffle(scenarios)
    
    # Re-assign IDs after shuffle
    for i, scenario in enumerate(scenarios):
        scenario['id'] = i
    
    return scenarios


def _legacy_generator_config(V, num_scenarios, seed):
    return {
        'V': int(V),
        'num_scenarios': int(num_scenarios),
        'seed': int(seed),
        'mixture': {'all_to_all_v': 0.5, 'allgather': 0.3, 'alltoall': 0.2},
        'all_to_all_v_range': [0, 4],
    }


def _config_hash(config):
    payload = json.dumps(config, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def load_or_generate_scenarios(
    scenario_path,
    V,
    num_scenarios,
    seed,
    generator_name=LEGACY_GENERATOR_NAME,
    generator_config=None,
):
    """
    Load scenarios from file if exists, otherwise generate and save.
    """
    expected_config = generator_config or _legacy_generator_config(V, num_scenarios, seed)
    expected_hash = _config_hash(expected_config)
    if os.path.exists(scenario_path):
        with open(scenario_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        metadata_matches = (
            isinstance(data, dict)
            and data.get('schema_version') == SCENARIO_SCHEMA_VERSION
            and data.get('generator_name') == generator_name
            and data.get('generator_config_hash') == expected_hash
            and isinstance(data.get('scenarios'), list)
        )
        if metadata_matches:
            return data['scenarios']

    scenarios = generate_scenarios(V, num_scenarios, seed)
    scenario_dir = os.path.dirname(scenario_path)
    if scenario_dir:
        os.makedirs(scenario_dir, exist_ok=True)
    with open(scenario_path, 'w', encoding='utf-8') as f:
        json.dump({
            'schema_version': SCENARIO_SCHEMA_VERSION,
            'generator_name': generator_name,
            'generator_config': expected_config,
            'generator_config_hash': expected_hash,
            'V': V,
            'num_scenarios': num_scenarios,
            'seed': seed,
            'scenarios': scenarios
        }, f, indent=2)
    return scenarios


def build_problem_from_scenario(V, E, edges, capacities, shared_constraints, scenario, T=20):
    """
    Build a ProblemInstance from a scenario dict.
    
    Args:
        V, E, edges, capacities, shared_constraints: Topology info
        scenario: Scenario dict with 'initial_state' and 'demands'
        T: Time limit
        
    Returns:
        ProblemInstance
    """
    C = scenario['C']
    initial_state = np.asarray(scenario['initial_state'], dtype=int).reshape(C, V)
    demands = np.asarray(scenario['demands'], dtype=int).reshape(C, V)
    
    metadata = {
        key: value for key, value in scenario.items()
        if key not in {'initial_state', 'demands', 'traffic_matrix'}
    }
    return ProblemInstance(
        V,
        C,
        E,
        T,
        capacities,
        edges,
        demands,
        initial_state,
        shared_constraints,
        traffic_matrix=scenario.get('traffic_matrix'),
        scenario_type=scenario.get('type'),
        sequence_id=scenario.get('sequence_id'),
        sequence_step=scenario.get('sequence_step'),
        metadata=metadata,
    )


# Get the directory where this evaluator.py file is located
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Go up to RLCCL root

def load_topology_info(topology_name):
    """
    Load topology info from a named topology directory.
    
    Args:
        topology_name: Name of the topology directory (e.g., 'Rear4GPU', 'Rear8GPU_NoSwitch_Test')
        
    Returns:
        TopologyInfo object
    """
    base_dir = _BASE_DIR
    json_path = f'{base_dir}/Data/{topology_name}/Topology/pipeline_topology_no_switch.json'
    cache_dir = f'{base_dir}/Data/{topology_name}/Topology'  # Cache in the same directory
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Topology file not found: {json_path}")
        
    V, E, edges, capacities, shared_constraints = load_topology_from_json(json_path)
    
    # Normalize capacities by the minimum bandwidth
    min_capacity = np.min(capacities)
    if min_capacity > 0:
        capacities = capacities / min_capacity
        shared_constraints = [(indices, limit / min_capacity) for indices, limit in shared_constraints]
    
    topology_info = TopologyInfo(
        V, E, edges, capacities, shared_constraints, cache_dir=cache_dir, name=topology_name
    )
    
    return topology_info


def generate_train_test_split(num_train=80, num_test=20, seed=42, 
                               topologies=None):
    """
    Generate training and test scenarios with different seeds to ensure no overlap.
    Supports multiple topologies for cross-topology generalization.
    
    Scenarios are cached to disk for reproducibility.
    
    Args:
        num_train: Number of training scenarios per topology
        num_test: Number of test scenarios per topology
        seed: Random seed
        topologies: List of topology names. If None, uses ['Rear4GPU', 'Rear8GPU_NoSwitch_Test']
        
    Returns:
        (train_problems, test_problems, topology_info_dict)
        - train_problems: List of (scenario_id, problem) tuples, each problem has its own topology_info
        - test_problems: List of (scenario_id, problem) tuples
        - topology_info_dict: Dict mapping topology_name -> TopologyInfo (for reference, but each problem carries its own)
    """
    if topologies is None:
        topologies = ['Rear4GPU', 'Rear8GPU_NoSwitch_Test', 'Heterogeneous_12GPU', 'Heterogeneous_16GPU_3Server', 'Heterogeneous_6GPU_Ring']
    
    base_dir = _BASE_DIR
    
    train_problems = []
    test_problems = []
    topology_info_dict = {}
    
    for topo_idx, topo_name in enumerate(topologies):
        # Load topology
        topology_info = load_topology_info(topo_name)
        topology_info_dict[topo_name] = topology_info
        
        V = topology_info.V
        E = topology_info.E
        edges = topology_info.edges
        capacities = topology_info.capacities
        shared_constraints = topology_info.shared_constraints
        
        # Use different seeds for different topologies to ensure diversity
        topo_seed = seed + topo_idx * 10000
        
        train_scenario_path = f'{base_dir}/Data/{topo_name}/scenarios_{LEGACY_GENERATOR_NAME}_train_{num_train}_seed{topo_seed}.json'
        test_scenario_path = f'{base_dir}/Data/{topo_name}/scenarios_{LEGACY_GENERATOR_NAME}_test_{num_test}_seed{topo_seed + 1000}.json'
        
        # Load or generate training scenarios
        train_scenarios = load_or_generate_scenarios(train_scenario_path, V, num_train, seed=topo_seed)
        
        # Load or generate test scenarios (different seed to ensure no overlap)
        test_scenarios = load_or_generate_scenarios(test_scenario_path, V, num_test, seed=topo_seed + 1000)
        
        # Time limit depends on problem size
        T = 30  # Increased for larger problems
        
        # Build problem instances with per-problem topology_info
        for scenario in train_scenarios:
            problem = build_problem_from_scenario(V, E, edges, capacities, shared_constraints, scenario, T)
            # Attach topology_info to problem
            problem.topology_info = topology_info
            problem.topology_name = topo_name
            # Use global unique ID
            global_id = f"{topo_name}_{scenario['id']}"
            train_problems.append((global_id, problem))
        
        for scenario in test_scenarios:
            problem = build_problem_from_scenario(V, E, edges, capacities, shared_constraints, scenario, T)
            # Attach topology_info to problem
            problem.topology_info = topology_info
            problem.topology_name = topo_name
            global_id = f"{topo_name}_{scenario['id']}"
            test_problems.append((global_id, problem))
    
    return train_problems, test_problems, topology_info_dict


def generate_train_test_split_single_topology(num_train=80, num_test=20, seed=42):
    """
    Legacy function: Generate training and test scenarios for a single topology.
    Kept for backward compatibility.
    
    Returns:
        (train_problems, test_problems, topology_info)
    """
    # Use relative paths based on this file's location
    base_dir = _BASE_DIR
    json_path = f'{base_dir}/Data/Rear8GPU_NoSwitch_Test/Topology/pipeline_topology_no_switch.json'
    train_scenario_path = f'{base_dir}/Data/scenarios_{LEGACY_GENERATOR_NAME}_train_{num_train}_seed{seed}.json'
    test_scenario_path = f'{base_dir}/Data/scenarios_{LEGACY_GENERATOR_NAME}_test_{num_test}_seed{seed + 1000}.json'
    
    if not os.path.exists(json_path):
        print(f"Warning: Topology file {json_path} not found.")
        return [], [], None
        
    V, E, edges, capacities, shared_constraints = load_topology_from_json(json_path)
    
    # Normalize capacities by the minimum bandwidth
    min_capacity = np.min(capacities)
    if min_capacity > 0:
        capacities = capacities / min_capacity
        shared_constraints = [(indices, limit / min_capacity) for indices, limit in shared_constraints]
    
    # Create topology info object
    topology_info = TopologyInfo(V, E, edges, capacities, shared_constraints)
    
    # Load or generate training scenarios
    train_scenarios = load_or_generate_scenarios(train_scenario_path, V, num_train, seed=seed)
    
    # Load or generate test scenarios (different seed to ensure no overlap)
    test_scenarios = load_or_generate_scenarios(test_scenario_path, V, num_test, seed=seed + 1000)
    
    # Time limit depends on problem size
    T = 30  # Increased for larger problems
    
    # Build problem instances with per-problem topology_info
    train_problems = []
    for scenario in train_scenarios:
        problem = build_problem_from_scenario(V, E, edges, capacities, shared_constraints, scenario, T)
        problem.topology_info = topology_info
        problem.topology_name = 'Rear8GPU_NoSwitch_Test'
        train_problems.append((scenario['id'], problem))
    
    test_problems = []
    for scenario in test_scenarios:
        problem = build_problem_from_scenario(V, E, edges, capacities, shared_constraints, scenario, T)
        problem.topology_info = topology_info
        problem.topology_name = 'Rear8GPU_NoSwitch_Test'
        test_problems.append((scenario['id'], problem))
    
    return train_problems, test_problems, topology_info


def generate_problems():
    """
    For backward compatibility: returns all problems (train + test).
    Use generate_train_test_split() for proper evaluation.
    """
    train_problems, test_problems, _ = generate_train_test_split()
    return train_problems + test_problems


def generate_problem():
    """
    For backward compatibility: returns the first problem.
    """
    problems = generate_problems()
    if problems:
        return problems[0][1]
    return None

def evaluate_schedule(schedule, problem):
    """
    Evaluates a schedule.
    schedule: List of T matrices, each (C, E) representing Y_t
    
    Returns:
        score: Combined score where:
            - Primary: negative completion step (fewer steps = higher score)
            - Secondary: weighted satisfaction as tiebreaker
            Score = -completion_step + 0.001 * weighted_score
            
        Error score: -(T + 2) to ensure errors rank below any valid completion
    """
    # Error score: worse than any valid completion (which is at most -T)
    error_score = -(problem.T + 2)
    
    if len(schedule) > problem.T:
        return error_score, "Schedule length exceeds time limit"
        
    current_state = problem.initial_state.copy() # (C, V) - who has what
    current_demands = problem.demands.copy()     # (C, V) - who wants what
    
    weighted_score = 0.0  # Tiebreaker score
    completion_step = len(schedule)  # Default to max if not completed
    
    for t, Y_t in enumerate(schedule):
        # Y_t should be (C, E)
        if Y_t.shape != (problem.C, problem.E):
            return error_score, f"Invalid shape for Y_{t}: {Y_t.shape}, expected ({problem.C}, {problem.E})"
            
        # 1. Bandwidth Constraint: sum_c Y_t(c, e) <= capacity[e]
        edge_loads = np.sum(Y_t, axis=0)
        if np.any(edge_loads > problem.capacities + 1e-6):
             idx = np.where(edge_loads > problem.capacities + 1e-6)[0][0]
             return error_score, f"Bandwidth constraint violated at step {t} on edge {idx} ({edge_loads[idx]} > {problem.capacities[idx]})"
             
        # 3. Shared Bandwidth Constraints
        for indices, limit in problem.shared_constraints:
            group_load = np.sum(edge_loads[indices])
            if group_load > limit + 1e-6:
                return error_score, f"Shared bandwidth constraint violated at step {t} (load {group_load} > {limit})"
            
        # 2. Feasibility Constraint: Source must have chunk
        for e in range(problem.E):
            u, v = problem.topology[e]
            chunks_on_edge = np.where(Y_t[:, e] == 1)[0]
            for c in chunks_on_edge:
                if current_state[c, u] == 0:
                    return error_score, f"Feasibility violated at step {t}: Node {u} sends chunk {c} but doesn't have it"
        
        # Calculate received chunks without the removed dense (E, V) D matrix.
        topology_info = getattr(problem, 'topology_info', None)
        if topology_info is not None and hasattr(topology_info, 'edge_dst'):
            edge_dst = np.asarray(topology_info.edge_dst, dtype=np.int64)
        else:
            edge_dst = np.asarray(problem.topology, dtype=np.int64)[:, 1]
        N_t = compute_received_chunks(Y_t, edge_dst, problem.V)
        
        # Calculate weighted score for tiebreaker
        satisfied = N_t * current_demands
        num_satisfied = np.sum(satisfied)
        
        w_t = 1.0 / (t + 1)
        weighted_score += w_t * num_satisfied
        
        # Update State
        current_state = np.maximum(current_state, N_t)
        current_demands = current_demands * (1 - N_t)
        
        if np.sum(current_demands) == 0:
            completion_step = t + 1  # Record completion step (1-indexed)
            break
    
    # Final score: primary = -completion_step, secondary = weighted_score as tiebreaker
    # Fewer steps = higher score (less negative)
    # Same steps: higher weighted_score = higher total score
    final_score = -completion_step + 0.001 * weighted_score
            
    return final_score, ""

def run_with_timeout(program_path, timeout_seconds=5000):
    """
    Run the user program in a separate subprocess.
    
    The program should implement:
    1. train(train_problems, topology_info) -> trained_model
    2. solve_with_model(problem, model, topology_info) -> schedule
    
    Evaluation:
    - Train on training set
    - Evaluate on test set (generalization)
    
    Args:
        program_path: Path to the program file
        timeout_seconds: Maximum execution time in seconds
        
    Returns:
        score (float) or (score, error_msg) tuple
    """
    evaluator_path = os.path.abspath(__file__)
    
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode='w') as temp_file:
        script = f'''
import sys
import os
import pickle
import traceback
import numpy as np

sys.path.insert(0, os.path.dirname('{evaluator_path}'))
sys.path.insert(0, os.path.dirname('{program_path}'))

results_path = '{temp_file.name}.results'

try:
    # Import the evaluator module
    import importlib.util
    spec = importlib.util.spec_from_file_location("evaluator_module", '{evaluator_path}')
    evaluator_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator_module)
    
    # Import the user program
    spec2 = importlib.util.spec_from_file_location("user_module", '{program_path}')
    user_module = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(user_module)
    
    # Generate train/test split (multi-topology)
    train_problems, test_problems, topology_info_dict = evaluator_module.generate_train_test_split(
        num_train=30, num_test=30, seed=42
    )
    
    # Check if the program has the new interface
    has_train = hasattr(user_module, "train")
    has_solve_with_model = hasattr(user_module, "solve_with_model")
    has_solve = hasattr(user_module, "solve")
    
    if has_train and has_solve_with_model:
        # NEW INTERFACE: Train-then-generalize paradigm (cross-topology)
        print("Using train-then-generalize interface (cross-topology)...")
        
        # Phase 1: Training (pass topology_info_dict for reference, but each problem has its own topology_info)
        print(f"Training on {{len(train_problems)}} scenarios across {{len(topology_info_dict)}} topologies...")
        model = user_module.train(train_problems, topology_info_dict)
        
        # Phase 2: Evaluation on TEST set (generalization across topologies)
        print(f"Evaluating on {{len(test_problems)}} test scenarios...")
        total_score = 0.0
        num_evaluated = 0
        errors = []
        
        for scenario_id, problem in test_problems:
            # Error score for this problem: -(T + 2)
            error_score = -(problem.T + 2)
            try:
                # Use problem's own topology_info for solving
                schedule = user_module.solve_with_model(problem, model, problem.topology_info)
                score, msg = evaluator_module.evaluate_schedule(schedule, problem)
                
                if msg:
                    errors.append(f"Test Scenario {{scenario_id}}: {{msg}}")
                    # score already contains error_score from evaluate_schedule
                
                total_score += score
                num_evaluated += 1
            except Exception as e:
                errors.append(f"Test Scenario {{scenario_id}}: {{str(e)}}")
                # Add error score for exceptions
                total_score += error_score
                num_evaluated += 1
        
        avg_score = total_score / max(1, num_evaluated)
        
        results = {{
            'score': float(avg_score),
            'total_score': float(total_score),
            'num_evaluated': num_evaluated,
            'num_test_scenarios': len(test_problems),
            'num_train_scenarios': len(train_problems),
            'interface': 'train-then-generalize',
            'error': "; ".join(errors[:5]) if errors else None
        }}
        
    else:
        # NO LEGACY INTERFACE ALLOWED!
        # Must implement train() and solve_with_model() for train-then-generalize paradigm
        missing = []
        if not has_train:
            missing.append("train(train_problems, topology_info)")
        if not has_solve_with_model:
            missing.append("solve_with_model(problem, model, topology_info)")
        
        raise AttributeError(
            f"REQUIRED: You MUST implement the train-then-generalize interface!\\n"
            f"Missing functions: {{', '.join(missing)}}\\n"
            f"\\n"
            f"Required interface:\\n"
            f"  def train(train_problems, topology_info) -> model\\n"
            f"  def solve_with_model(problem, model, topology_info) -> schedule\\n"
            f"\\n"
            f"Per-instance solving (solve(problem)) is NOT allowed!"
        )
    
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
        
except Exception as e:
    tb = traceback.format_exc()
    # Use a very negative score for errors (worse than any valid completion)
    # Since T=30, error_score = -(T+2) = -32, we use -100 to be safe
    results = {{'score': -100.0, 'error': f'{{e}}', 'traceback': tb}}
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    sys.exit(0)
'''
        temp_file.write(script)
        temp_file_path = temp_file.name
    
    results_path = f"{temp_file_path}.results"
    
    try:
        # Run the script in subprocess
        process = subprocess.Popen(
            [sys.executable, temp_file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            
            # Error score constant (worse than any valid completion with T=30)
            ERROR_SCORE = -100.0
            
            if os.path.exists(results_path):
                with open(results_path, 'rb') as f:
                    results = pickle.load(f)
                
                if results.get('error'):
                    error_msg = results.get('error', '')
                    if 'traceback' in results:
                        error_msg += f"\\nTraceback:\\n{results['traceback']}"
                    return results.get('score', ERROR_SCORE), error_msg
                return results.get('score', 0.0)
            else:
                # No results file
                return ERROR_SCORE, f"No results file. stdout: {stdout.decode()}, stderr: {stderr.decode()}"
                
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            return ERROR_SCORE, f"Process timed out after {timeout_seconds} seconds"
            
    finally:
        # Cleanup temp files
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        if os.path.exists(results_path):
            os.unlink(results_path)

def evaluate(path_user_py, gpu_rank=None):
    """
    Evaluate the program by running it and checking the score.
    
    Args:
        program_path: Path to the program file
        gpu_rank: GPU ID to use for this evaluation (optional)
        
    Returns:
        Dictionary of metrics
        
    Score interpretation:
        - Valid scores: negative values where less negative = better
          (e.g., -5 means completed in 5 steps, better than -10)
        - Error score: -100.0 (worse than any valid completion with T=30)
    """
    # Set CUDA_VISIBLE_DEVICES if gpu_rank is provided
    if gpu_rank is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_rank)
    
    # Error score constant (worse than any valid completion with T=30)
    ERROR_SCORE = -100.0
    
    program_path = os.path.abspath(path_user_py)
    try:
        result = run_with_timeout(program_path)
        
        score = 0.0
        error_info = None
        
        if isinstance(result, tuple):
            score = result[0]
            error_info = result[1]
        else:
            score = result
        
        # If there's an error message, the score from run_with_timeout is already
        # set to ERROR_SCORE (-100.0), so we just pass it through
        # No need to check for specific values like -1.0 anymore
                
        return {
            "score": float(score),
            "combined_score": float(score),
            "error_info": {"run_error": error_info} if error_info else {}
        }
        
    except Exception as e:
        return {
            "score": ERROR_SCORE,
            "combined_score": ERROR_SCORE,
            "error_info": {"exception": str(e)}
        }

if __name__ == "__main__":


    generate_problem()

    if len(sys.argv) < 2:
        print("Usage: python evaluator.py <program_path>")
        sys.exit(1)
        
    program_path = sys.argv[1]
    result = evaluate(program_path)
    print(result)

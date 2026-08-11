#!/usr/bin/env python3
"""
灵活的单问题测试脚本
支持指定不同的collective类型(AllGather, AllToAll等)和chunk factor
生成人类可读的策略文件和MSCCL XML文件
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import torch

# 添加路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from rlccl.models.slot_policy import SlotLevelPolicy
from rlccl.envs.decoder import SlotDecoder
from rlccl.envs.evaluator import evaluate_schedule, load_topology_info
from rlccl.envs.problem import compute_received_chunks
from rlccl.utils.xml_converter import tensors_to_msccl_xml


def generate_allgather_problem(topology_info, chunk_factor, T=30):
    """生成AllGather问题实例
    
    每个节点有chunk_factor个chunk，需要gather到所有其他节点
    """
    V = topology_info.V
    E = topology_info.E
    compute_nodes = list(range(V))
    num_compute = len(compute_nodes)
    
    total_chunks = num_compute * chunk_factor
    C = total_chunks
    
    initial_state = np.zeros((C, V), dtype=int)
    demands = np.zeros((C, V), dtype=int)
    
    for chunk_idx in range(total_chunks):
        src_idx = chunk_idx // chunk_factor
        src_node = compute_nodes[src_idx]
        initial_state[chunk_idx, src_node] = 1
        
        # 每个chunk需要分发到所有其他节点
        for dst_node in compute_nodes:
            if dst_node != src_node:
                demands[chunk_idx, dst_node] = 1
    
    from rlccl.envs.evaluator import ProblemInstance
    problem = ProblemInstance(
        num_nodes=V,
        num_chunks=C,
        num_edges=E,
        time_limit=T,
        capacities=topology_info.capacities,
        topology=topology_info.edges,
        demands=demands,
        initial_state=initial_state,
        shared_constraints=topology_info.shared_constraints,
        topology_info=topology_info
    )
    
    return problem, "AllGather"


def generate_alltoall_problem(topology_info, chunk_factor, T=30):
    """生成AllToAll问题实例
    
    每个节点有N*chunk_factor个chunk，每个节点需要接收来自其他N-1个节点的chunk
    """
    V = topology_info.V
    E = topology_info.E
    compute_nodes = list(range(V))
    num_compute = len(compute_nodes)
    
    # AllToAll: 每个节点有 num_compute * chunk_factor 个chunk
    total_chunks = num_compute * num_compute * chunk_factor
    C = total_chunks
    
    initial_state = np.zeros((C, V), dtype=int)
    demands = np.zeros((C, V), dtype=int)
    
    for src_idx in range(num_compute):
        src_node = compute_nodes[src_idx]
        for dst_idx in range(num_compute):
            dst_node = compute_nodes[dst_idx]
            # 从src_node到dst_node的chunks
            for cf in range(chunk_factor):
                chunk_id = (src_idx * num_compute + dst_idx) * chunk_factor + cf
                initial_state[chunk_id, src_node] = 1
                if src_node != dst_node:
                    demands[chunk_id, dst_node] = 1
    
    from rlccl.envs.evaluator import ProblemInstance
    problem = ProblemInstance(
        num_nodes=V,
        num_chunks=C,
        num_edges=E,
        time_limit=T,
        capacities=topology_info.capacities,
        topology=topology_info.edges,
        demands=demands,
        initial_state=initial_state,
        shared_constraints=topology_info.shared_constraints,
        topology_info=topology_info
    )
    
    return problem, "AllToAll"


def generate_allreduce_problem(topology_info, chunk_factor, T=30):
    """生成AllReduce问题实例（简化版，作为AllGather + ReduceScatter）
    
    每个节点有chunk_factor个chunk，需要reduce并分发到所有节点
    简化为：每个节点的chunk需要到达所有节点
    """
    V = topology_info.V
    E = topology_info.E
    compute_nodes = list(range(V))
    num_compute = len(compute_nodes)
    
    total_chunks = num_compute * chunk_factor
    C = total_chunks
    
    initial_state = np.zeros((C, V), dtype=int)
    demands = np.zeros((C, V), dtype=int)
    
    for chunk_idx in range(total_chunks):
        src_idx = chunk_idx // chunk_factor
        src_node = compute_nodes[src_idx]
        initial_state[chunk_idx, src_node] = 1
        
        # AllReduce: 每个chunk最终需要在所有节点上
        for dst_node in compute_nodes:
            if dst_node != src_node:
                demands[chunk_idx, dst_node] = 1
    
    from rlccl.envs.evaluator import ProblemInstance
    problem = ProblemInstance(
        num_nodes=V,
        num_chunks=C,
        num_edges=E,
        time_limit=T,
        capacities=topology_info.capacities,
        topology=topology_info.edges,
        demands=demands,
        initial_state=initial_state,
        shared_constraints=topology_info.shared_constraints,
        topology_info=topology_info
    )
    
    return problem, "AllReduce"


def generate_broadcast_problem(topology_info, chunk_factor, root_node=0, T=30):
    """生成Broadcast问题实例
    
    root节点有chunk_factor个chunk，需要广播到所有其他节点
    """
    V = topology_info.V
    E = topology_info.E
    compute_nodes = list(range(V))
    
    C = chunk_factor
    
    initial_state = np.zeros((C, V), dtype=int)
    demands = np.zeros((C, V), dtype=int)
    
    # Root节点拥有所有chunks
    initial_state[:, root_node] = 1
    
    # 所有其他节点都需要这些chunks
    for dst_node in compute_nodes:
        if dst_node != root_node:
            demands[:, dst_node] = 1
    
    from rlccl.envs.evaluator import ProblemInstance
    problem = ProblemInstance(
        num_nodes=V,
        num_chunks=C,
        num_edges=E,
        time_limit=T,
        capacities=topology_info.capacities,
        topology=topology_info.edges,
        demands=demands,
        initial_state=initial_state,
        shared_constraints=topology_info.shared_constraints,
        topology_info=topology_info
    )
    
    return problem, "Broadcast"


def solve_with_model(problem, model, topology_info, device, verbose=True):
    """使用模型求解问题"""
    model.eval()
    decoder = SlotDecoder(topology_info)
    
    state = problem.initial_state.copy()
    demands = problem.demands.copy()
    schedule = []
    
    if verbose:
        print(f"\n开始求解...")
        print(f"  初始状态: {np.sum(problem.initial_state)} chunks已分配")
        print(f"  需求总数: {np.sum(problem.demands)} 次传输需求")
    
    with torch.no_grad():
        for t in range(problem.T):
            Y_t, _, _, _, _, _ = decoder.decode_slot(
                model, state, demands, t, problem.T, train=False
            )
            schedule.append(Y_t)
            
            # 更新状态
            N_t = compute_received_chunks(Y_t, topology_info.edge_dst, topology_info.V)
            state = np.maximum(state, N_t)
            demands = demands * (1 - N_t)
            
            remaining = np.sum(demands)
            transfers = np.sum(Y_t)
            
            if verbose and transfers > 0:
                print(f"  步 {t}: {int(transfers)} 次传输, 剩余需求: {int(remaining)}")
            
            if remaining == 0:
                if verbose:
                    print(f"\n✅ 在第 {t+1} 步完成所有传输！")
                break
    
    # 填充剩余时间步
    while len(schedule) < problem.T:
        schedule.append(np.zeros((problem.C, problem.E), dtype=int))
    
    return schedule


def schedule_to_human_readable(schedule, problem, topology_info, collective_type, chunk_factor):
    """将调度策略转换为人类可读的格式"""
    output = []
    output.append("=" * 80)
    output.append(f"集合通信调度策略 - {collective_type}")
    output.append("=" * 80)
    output.append(f"\n拓扑: {getattr(topology_info, 'name', 'Unknown')}")
    output.append(f"节点数: {problem.V}")
    output.append(f"边数: {problem.E}")
    output.append(f"Chunk数: {problem.C}")
    output.append(f"Chunk Factor: {chunk_factor}")
    output.append(f"时间上限: {problem.T}")
    
    actual_steps = sum(1 for Y in schedule if np.sum(Y) > 0)
    output.append(f"实际完成步数: {actual_steps}")
    
    output.append("\n边定义:")
    for e, (u, v) in enumerate(problem.topology):
        cap = int(problem.capacities[e])
        output.append(f"  边 {e}: Node{u} -> Node{v} (容量={cap})")
    
    output.append("\n" + "-" * 80)
    output.append("调度详情 (每个时间步)")
    output.append("-" * 80)
    
    for t, Y_t in enumerate(schedule):
        transfers = []
        for c in range(problem.C):
            for e in range(problem.E):
                if Y_t[c, e] == 1:
                    u, v = problem.topology[e]
                    transfers.append((c, u, v, e))
        
        if transfers:
            output.append(f"\n时间步 {t}:")
            
            # 按源节点分组
            by_src = {}
            for c, u, v, e in transfers:
                if u not in by_src:
                    by_src[u] = []
                by_src[u].append((c, v, e))
            
            for src in sorted(by_src.keys()):
                sends = by_src[src]
                output.append(f"  Node{src} 发送:")
                for c, dst, e in sends:
                    output.append(f"    Chunk {c} -> Node{dst} (边 {e})")
    
    # 统计信息
    output.append("\n" + "=" * 80)
    output.append("统计信息")
    output.append("=" * 80)
    
    total_transfers = sum(np.sum(Y) for Y in schedule)
    output.append(f"总传输数: {int(total_transfers)}")
    
    # 每个时间步的传输数
    output.append("\n每时间步传输数:")
    for t, Y_t in enumerate(schedule):
        count = np.sum(Y_t)
        if count > 0:
            output.append(f"  步 {t}: {int(count)} 次传输")
    
    # 边利用率
    output.append("\n边利用率:")
    edge_total = np.zeros(problem.E)
    for Y_t in schedule:
        edge_total += np.sum(Y_t, axis=0)
    
    # 按使用次数排序
    edge_usage = [(e, int(edge_total[e])) for e in range(problem.E)]
    edge_usage.sort(key=lambda x: x[1], reverse=True)
    
    max_display = min(20, len([e for e, c in edge_usage if c > 0]))
    output.append(f"\n高使用率边 (top {max_display}):")
    for e, count in edge_usage[:max_display]:
        if count > 0:
            u, v = problem.topology[e]
            output.append(f"  边 {e} (Node{u}->Node{v}): {count} 次使用")
    
    return "\n".join(output)


def export_to_xml(schedule, problem, topology_info, collective_type, output_path, instances=1):
    """导出为MSCCL XML格式（不依赖aiccl）。

    注意：当前 `rlccl.utils.xml_converter` 原生支持：AllGather / AllToAll。
    """
    try:
        # 转换为torch tensors
        real_strategy_list = [torch.from_numpy(Y_t).unsqueeze(0).float() for Y_t in schedule]
        pre_condition = torch.from_numpy(problem.initial_state).unsqueeze(0).float()
        edge_src_idx = torch.from_numpy(topology_info.edges[:, 0]).long()
        edge_dst_idx = torch.from_numpy(topology_info.edges[:, 1]).long()

        # 当前RLCCL TopologyInfo里没有显式switch标记，这里默认全部为GPU节点
        is_switch = torch.zeros(problem.V, dtype=torch.bool)
        chunk_mask = torch.ones(1, problem.C, dtype=torch.bool)

        # xml_converter的collective_type使用小写: allgather / alltoall
        ct = collective_type.strip().lower()
        if ct == "allgather":
            post_condition = None
        elif ct == "alltoall":
            # AllToAll: 每个chunk必须有且仅有一个目标GPU。
            # 对于demand为空的chunk(即src==dst的自发自收)，目标就是源节点。
            post = np.zeros((problem.C, problem.V), dtype=int)
            for c in range(problem.C):
                dsts = np.where(problem.demands[c] > 0)[0]
                if len(dsts) == 1:
                    post[c, int(dsts[0])] = 1
                elif len(dsts) == 0:
                    srcs = np.where(problem.initial_state[c] > 0)[0]
                    if len(srcs) != 1:
                        raise ValueError(f"AllToAll: chunk {c} invalid initial_state owners={srcs.tolist()}")
                    post[c, int(srcs[0])] = 1
                else:
                    raise ValueError(f"AllToAll: chunk {c} has multiple destinations={dsts.tolist()}")
            post_condition = torch.from_numpy(post).unsqueeze(0).float()
        else:
            raise ValueError(
                f"XML导出当前仅支持AllGather/AllToAll，collective_type={collective_type!r}"
            )

        xml_text = tensors_to_msccl_xml(
            real_strategy_list=real_strategy_list,
            pre_condition=pre_condition,
            post_condition=post_condition,
            edge_src_idx=edge_src_idx,
            edge_dst_idx=edge_dst_idx,
            is_switch=is_switch,
            chunk_mask=chunk_mask,
            program_name=f"rlccl_generated_{ct}",
            topology_mode="from-transfers",
            do_check=True,
            collective_type=ct,
            instances=instances,
            instr_fusion=False,
        )

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(xml_text)

        return True
    except Exception as e:
        print(f"⚠️  XML生成失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="测试单个问题实例")
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型路径")
    parser.add_argument("--topology", type=str, required=True,
                        help="拓扑名称 (如 Rear8GPU_NoSwitch_Test)")
    parser.add_argument("--collective", type=str, required=True,
                        choices=["allgather", "alltoall", "allreduce", "broadcast"],
                        help="集合通信类型")
    parser.add_argument("--chunk_factor", type=int, required=True,
                        help="Chunk factor")
    parser.add_argument("--time_limit", type=int, default=30,
                        help="时间上限")
    parser.add_argument("--root_node", type=int, default=0,
                        help="Broadcast的root节点 (仅用于broadcast)")
    parser.add_argument("--output_dir", type=str, default="./test_results",
                        help="输出目录")
    parser.add_argument("--export_xml", action="store_true",
                        help="导出MSCCL XML文件")
    parser.add_argument("--xml_instances", type=int, default=1,
                        help="MSCCL XML instances数量")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="设备 (cuda:0, cpu等)")
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="模型hidden dimension")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置设备
    if args.device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")
    
    print(f"{'='*80}")
    print("RLCCL 单问题测试")
    print(f"{'='*80}")
    print(f"模型: {args.model_path}")
    print(f"拓扑: {args.topology}")
    print(f"集合通信: {args.collective.upper()}")
    print(f"Chunk Factor: {args.chunk_factor}")
    print(f"时间上限: {args.time_limit}")
    print(f"设备: {device}")
    
    # 加载模型
    print(f"\n加载模型...")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    model = SlotLevelPolicy(
        node_feat_dim=5,
        edge_feat_dim=2,
        cand_feat_dim=5,
        chunk_feat_dim=2,
        hidden_dim=args.hidden_dim
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("✅ 模型加载成功")
    
    # 加载拓扑
    print(f"\n加载拓扑...")
    topology_info = load_topology_info(args.topology)
    print(f"  节点数: {topology_info.V}")
    print(f"  边数: {topology_info.E}")
    
    # 生成问题
    print(f"\n生成问题实例...")
    if args.collective == "allgather":
        problem, collective_type = generate_allgather_problem(
            topology_info, args.chunk_factor, args.time_limit
        )
    elif args.collective == "alltoall":
        problem, collective_type = generate_alltoall_problem(
            topology_info, args.chunk_factor, args.time_limit
        )
    elif args.collective == "allreduce":
        problem, collective_type = generate_allreduce_problem(
            topology_info, args.chunk_factor, args.time_limit
        )
    elif args.collective == "broadcast":
        problem, collective_type = generate_broadcast_problem(
            topology_info, args.chunk_factor, args.root_node, args.time_limit
        )
    else:
        raise ValueError(f"Unsupported collective: {args.collective}")
    
    print(f"  总Chunk数: {problem.C}")
    print(f"  总需求数: {np.sum(problem.demands)}")
    
    # 使用模型求解
    schedule = solve_with_model(problem, model, topology_info, device)
    
    # 评估结果
    print(f"\n{'='*80}")
    print("评估结果")
    print(f"{'='*80}")
    
    score, error_msg = evaluate_schedule(schedule, problem)
    
    if error_msg:
        print(f"❌ 错误: {error_msg}")
        print(f"   分数: {score}")
    else:
        completion_steps = -int(score // 1)
        weighted_score = (score + completion_steps) / 0.001
        
        print(f"✅ 测试成功！")
        print(f"   完成步数: {completion_steps}")
        print(f"   总分数: {score:.4f}")
        print(f"   加权分数: {weighted_score:.2f}")
        
        total_transfers = sum(np.sum(Y) for Y in schedule if np.sum(Y) > 0)
        print(f"\n详细统计:")
        print(f"   总传输次数: {int(total_transfers)}")
        if completion_steps > 0:
            print(f"   平均每步传输: {total_transfers / completion_steps:.1f}")
    
    # 生成人类可读的策略文件
    print(f"\n生成策略文件...")
    readable_strategy = schedule_to_human_readable(
        schedule, problem, topology_info, collective_type, args.chunk_factor
    )
    
    strategy_file = os.path.join(
        args.output_dir,
        f"strategy_{args.topology}_{collective_type}_cf{args.chunk_factor}.txt"
    )
    with open(strategy_file, 'w', encoding='utf-8') as f:
        f.write(readable_strategy)
    
    print(f"✅ 策略文件: {strategy_file}")
    
    # 生成XML文件
    if args.export_xml:
        print(f"\n生成MSCCL XML文件...")
        xml_file = os.path.join(
            args.output_dir,
            f"{args.topology}_{collective_type.lower()}_cf{args.chunk_factor}_inst{args.xml_instances}.xml"
        )
        if export_to_xml(schedule, problem, topology_info, collective_type, xml_file, args.xml_instances):
            print(f"✅ XML文件: {xml_file}")
    
    # 保存摘要
    summary_file = os.path.join(args.output_dir, "test_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"测试结果摘要\n")
        f.write(f"{'='*80}\n")
        f.write(f"模型: {args.model_path}\n")
        f.write(f"拓扑: {args.topology}\n")
        f.write(f"集合通信: {collective_type}\n")
        f.write(f"Chunk Factor: {args.chunk_factor}\n")
        f.write(f"时间上限: {args.time_limit}\n")
        f.write(f"\n结果:\n")
        if error_msg:
            f.write(f"  状态: 失败\n")
            f.write(f"  错误: {error_msg}\n")
        else:
            f.write(f"  状态: 成功\n")
            f.write(f"  完成步数: {completion_steps}\n")
            f.write(f"  总分数: {score:.4f}\n")
            f.write(f"  总传输次数: {int(total_transfers)}\n")
        f.write(f"\n测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"✅ 摘要文件: {summary_file}")
    print(f"\n✅ 测试完成！")


if __name__ == "__main__":
    main()

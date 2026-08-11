#!/usr/bin/env python3
"""Create topology variants where cross-server IB links only connect GPUs
that share the same NIC group:

- NIC0 GPUs (e.g., GPU 0,1 per server) can only connect to NIC0 GPUs on other servers
- NIC1 GPUs (e.g., GPU 2,3 per server) can only connect to NIC1 GPUs on other servers
- No cross-NIC connections between servers (e.g., server1:GPU0 cannot connect to server2:GPU2)

This script reads existing topology JSONs under RLCCL/Data/<topology>/Topology/
then writes new topology folders under RLCCL/Data/<new_topology>/Topology/.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional


_MLX5_GROUP_RE = re.compile(r"^(server\d+)_mlx5_(\d+)_(egress|ingress)$")


@dataclass(frozen=True)
class TopologyPaths:
    src_json: Path
    dst_json: Path


def _server_from_physical(physical: str) -> str:
    # e.g. "server4:GPU2" -> "server4"
    return physical.split(":", 1)[0]


def _gpu_index_from_physical(physical: str) -> int:
    # e.g. "server4:GPU2" -> 2
    m = re.search(r"GPU(\d+)", physical)
    return int(m.group(1)) if m else -1


def _build_id_to_server(nodes: List[dict]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for n in nodes:
        nid = int(n["id"])
        out[nid] = _server_from_physical(n.get("physical", ""))
    return out


def _build_id_to_gpu_index(nodes: List[dict]) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for n in nodes:
        nid = int(n["id"])
        out[nid] = n.get("gpu_index", _gpu_index_from_physical(n.get("physical", "")))
    return out


def _infer_nic_ports(bandwidth_groups: Dict[str, dict]) -> Set[int]:
    """Get all mlx5 port numbers used."""
    ports = set()
    for k in bandwidth_groups.keys():
        m = _MLX5_GROUP_RE.match(k)
        if m:
            ports.add(int(m.group(2)))
    return ports


def _build_gpu_to_nic_port(nodes: List[dict], bandwidth_groups: Dict[str, dict]) -> Dict[int, int]:
    """Map each GPU node id to its NIC port (mlx5_X number).
    
    We infer this from bandwidth_groups: if a GPU appears in server*_mlx5_X_egress,
    it uses NIC port X.
    """
    gpu_to_port: Dict[int, int] = {}
    
    for gname, g in bandwidth_groups.items():
        m = _MLX5_GROUP_RE.match(gname)
        if not m:
            continue
        
        server_name, port_s, direction = m.group(1), m.group(2), m.group(3)
        port = int(port_s)
        
        # Only look at egress groups to avoid double-counting
        if direction != "egress":
            continue
            
        for e in g.get("edges", []) or []:
            src = int(e["source"])
            # The source GPU uses this NIC port
            if src not in gpu_to_port:
                gpu_to_port[src] = port
    
    return gpu_to_port


def transform_topology_same_nic(data: dict) -> dict:
    """Transform topology so cross-server IB only connects same-NIC GPUs.
    
    E.g., if GPU 0,1 share NIC0 and GPU 2,3 share NIC1:
    - GPU0 can connect to GPU0,1 on other servers
    - GPU2 can connect to GPU2,3 on other servers
    """
    nodes = data.get("nodes", [])
    links = data.get("links", [])
    bandwidth_groups = data.get("bandwidth_groups", {}) or {}

    id_to_server = _build_id_to_server(nodes)
    id_to_gpu_idx = _build_id_to_gpu_index(nodes)
    gpu_to_nic = _build_gpu_to_nic_port(nodes, bandwidth_groups)
    nic_ports = _infer_nic_ports(bandwidth_groups)
    
    # Helper: check if two GPUs can connect (same NIC port)
    def same_nic(src_id: int, dst_id: int) -> bool:
        src_nic = gpu_to_nic.get(src_id)
        dst_nic = gpu_to_nic.get(dst_id)
        if src_nic is None or dst_nic is None:
            # If we can't determine NIC, keep the link
            return True
        return src_nic == dst_nic

    # 1) Filter IB links: only keep cross-server links where both GPUs use same NIC
    new_links: List[dict] = []
    dropped = 0
    kept_ib_edges: Set[Tuple[int, int]] = set()
    
    for link in links:
        if link.get("type") != "ib":
            new_links.append(link)
            continue

        src = int(link["source"])
        dst = int(link["target"])
        src_srv = id_to_server.get(src)
        dst_srv = id_to_server.get(dst)
        
        # Same server: keep (shouldn't happen for IB, but just in case)
        if src_srv == dst_srv:
            new_links.append(link)
            kept_ib_edges.add((src, dst))
            continue
        
        # Cross-server: only keep if same NIC
        if same_nic(src, dst):
            new_links.append(link)
            kept_ib_edges.add((src, dst))
        else:
            dropped += 1

    # 2) Filter bandwidth_groups: only keep edges that still exist
    new_groups: Dict[str, dict] = {}
    for gname, g in bandwidth_groups.items():
        m = _MLX5_GROUP_RE.match(gname)
        if not m:
            # Non-mlx5 groups: keep as-is
            new_groups[gname] = g
            continue

        edges = g.get("edges", []) or []
        kept_edges: List[dict] = []
        for e in edges:
            src = int(e["source"])
            dst = int(e["target"])
            if (src, dst) in kept_ib_edges:
                kept_edges.append(e)

        if kept_edges:
            g2 = dict(g)
            g2["edges"] = kept_edges
            if isinstance(g2.get("description"), str):
                g2["description"] = g2["description"] + " | same-NIC only"
            new_groups[gname] = g2
        # else: drop empty groups

    out = dict(data)
    out["links"] = new_links
    out["bandwidth_groups"] = new_groups

    # Add metadata for traceability
    graph_meta = dict(out.get("graph") or {})
    graph_meta["same_nic_connect"] = {
        "rule": "cross-server IB only between GPUs sharing same NIC port",
        "nic_ports": sorted(nic_ports),
        "dropped_ib_links": dropped,
    }
    out["graph"] = graph_meta
    return out


def transform_topology_per_gpu_nic(data: dict) -> dict:
    """Transform topology so each GPU has its own NIC.
    
    Cross-server IB only connects GPUs with the same gpu_index:
    - GPU0 can only connect to GPU0 on other servers
    - GPU1 can only connect to GPU1 on other servers
    - etc.
    """
    nodes = data.get("nodes", [])
    links = data.get("links", [])
    bandwidth_groups = data.get("bandwidth_groups", {}) or {}

    id_to_server = _build_id_to_server(nodes)
    id_to_gpu_idx = _build_id_to_gpu_index(nodes)
    
    # 1) Filter IB links: only keep cross-server links where gpu_index matches
    new_links: List[dict] = []
    dropped = 0
    kept_ib_edges: Set[Tuple[int, int]] = set()
    
    for link in links:
        if link.get("type") != "ib":
            new_links.append(link)
            continue

        src = int(link["source"])
        dst = int(link["target"])
        src_srv = id_to_server.get(src)
        dst_srv = id_to_server.get(dst)
        src_gpu = id_to_gpu_idx.get(src)
        dst_gpu = id_to_gpu_idx.get(dst)
        
        # Same server: keep
        if src_srv == dst_srv:
            new_links.append(link)
            kept_ib_edges.add((src, dst))
            continue
        
        # Cross-server: only keep if same gpu_index
        if src_gpu == dst_gpu:
            new_links.append(link)
            kept_ib_edges.add((src, dst))
        else:
            dropped += 1

    # 2) Create new bandwidth_groups: one per GPU index
    # Each GPU now has its own NIC, so we create per-gpu-index groups
    new_groups: Dict[str, dict] = {}
    
    # Keep non-mlx5 groups (like nvswitch groups)
    for gname, g in bandwidth_groups.items():
        m = _MLX5_GROUP_RE.match(gname)
        if not m:
            new_groups[gname] = g
    
    # Build per-server, per-gpu-index groups
    # Group edges by (server, gpu_index, direction)
    from collections import defaultdict
    
    # First, find the bandwidth for each original mlx5 group
    # We'll use the minimum bandwidth from the original groups as a base
    min_bw = float('inf')
    bw_unit = "GBps"
    for gname, g in bandwidth_groups.items():
        m = _MLX5_GROUP_RE.match(gname)
        if m:
            bw = g.get("max_bandwidth", float('inf'))
            if bw < min_bw:
                min_bw = bw
                bw_unit = g.get("bandwidth_unit", "GBps")
    
    if min_bw == float('inf'):
        min_bw = 6.25  # Default fallback
    
    # Group kept IB edges by (src_server, src_gpu_idx) for egress
    # and by (dst_server, dst_gpu_idx) for ingress
    egress_edges: Dict[Tuple[str, int], List[dict]] = defaultdict(list)
    ingress_edges: Dict[Tuple[str, int], List[dict]] = defaultdict(list)
    
    for (src, dst) in kept_ib_edges:
        src_srv = id_to_server.get(src)
        dst_srv = id_to_server.get(dst)
        src_gpu = id_to_gpu_idx.get(src)
        dst_gpu = id_to_gpu_idx.get(dst)
        
        # Only cross-server edges need bandwidth groups
        if src_srv != dst_srv:
            egress_edges[(src_srv, src_gpu)].append({"source": src, "target": dst})
            ingress_edges[(dst_srv, dst_gpu)].append({"source": src, "target": dst})
    
    # Create egress groups
    for (srv, gpu_idx), edges in egress_edges.items():
        gname = f"{srv}_gpu{gpu_idx}_nic_egress"
        new_groups[gname] = {
            "max_bandwidth": min_bw,
            "bandwidth_unit": bw_unit,
            "description": f"Per-GPU NIC egress for {srv} GPU{gpu_idx}",
            "edges": edges
        }
    
    # Create ingress groups
    for (srv, gpu_idx), edges in ingress_edges.items():
        gname = f"{srv}_gpu{gpu_idx}_nic_ingress"
        new_groups[gname] = {
            "max_bandwidth": min_bw,
            "bandwidth_unit": bw_unit,
            "description": f"Per-GPU NIC ingress for {srv} GPU{gpu_idx}",
            "edges": edges
        }

    out = dict(data)
    out["links"] = new_links
    out["bandwidth_groups"] = new_groups

    # Add metadata
    graph_meta = dict(out.get("graph") or {})
    graph_meta["per_gpu_nic"] = {
        "rule": "cross-server IB only between GPUs with same gpu_index (1 NIC per GPU)",
        "dropped_ib_links": dropped,
    }
    out["graph"] = graph_meta
    return out


def create_variant(*, base_dir: Path, src_topology: str, dst_topology: str, 
                   transform_fn) -> TopologyPaths:
    src_json = base_dir / "Data" / src_topology / "Topology" / "pipeline_topology_no_switch.json"
    dst_json = base_dir / "Data" / dst_topology / "Topology" / "pipeline_topology_no_switch.json"
    dst_json.parent.mkdir(parents=True, exist_ok=True)

    with src_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    new_data = transform_fn(data)

    with dst_json.open("w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return TopologyPaths(src_json=src_json, dst_json=dst_json)


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]  # RLCCL/

    # Jobs: (src_topology, dst_topology, transform_function)
    jobs = [
        # Same-NIC connect: GPUs sharing a NIC can connect across servers
        (
            "Rear8GPU_NoSwitch_Test",
            "Rear8GPU_NoSwitch_Test_SameNICConnect",
            transform_topology_same_nic,
        ),
        (
            "Heterogeneous_32GPU_4Server",
            "Heterogeneous_32GPU_4Server_SameNICConnect",
            transform_topology_same_nic,
        ),
        (
            "Heterogeneous_64GPU_8Server",
            "Heterogeneous_64GPU_8Server_SameNICConnect",
            transform_topology_same_nic,
        ),
        # Per-GPU NIC: each GPU has its own NIC, only same gpu_index connects
        (
            "Heterogeneous_32GPU_4Server",
            "Heterogeneous_32GPU_4Server_PerGPU_NIC",
            transform_topology_per_gpu_nic,
        ),
    ]

    results: List[TopologyPaths] = []
    for src, dst, transform_fn in jobs:
        p = create_variant(base_dir=base_dir, src_topology=src, dst_topology=dst,
                          transform_fn=transform_fn)
        results.append(p)

    for p in results:
        print(f"Wrote: {p.dst_json} (from {p.src_json})")


if __name__ == "__main__":
    main()

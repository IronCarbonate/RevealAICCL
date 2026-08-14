"""Serialize the CPU static plan into one pointer-free GPU-friendly blob."""

from __future__ import annotations

from dataclasses import dataclass
import struct

from .scheduler_schema import ActionFlags, CompiledRouteTemplate, SchedulerConfig


PLAN_MAGIC = 0x524C43434C4D3401
PLAN_VERSION = 1
_HEADER = struct.Struct("<QIIQQIIIIIIQQQQQQ")
_ROUTE = struct.Struct("<iiiiQQII")
_I32 = struct.Struct("<i")
_U64 = struct.Struct("<Q")
_U32 = struct.Struct("<I")
_CHECKSUM_OFFSET = 24


def fnv1a64(blob: bytes, *, zero_checksum_field: bool = False) -> int:
    value = 0xCBF29CE484222325
    for index, byte in enumerate(blob):
        if zero_checksum_field and _CHECKSUM_OFFSET <= index < _CHECKSUM_OFFSET + 8:
            byte = 0
        value ^= byte
        value = (value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return value


@dataclass(frozen=True, slots=True)
class CompiledPlanBlob:
    config: SchedulerConfig
    route_templates: tuple[CompiledRouteTemplate, ...]
    rank_pair_to_route: tuple[int, ...]
    capacity_table: tuple[int, ...]
    legality_flags: tuple[int, ...]
    descriptor_stride: int
    region_bytes: int
    data: bytes
    checksum: int


def compile_rank_pair_plan(config: SchedulerConfig) -> CompiledPlanBlob:
    """Compile deterministic direct rank-pair lookup used by R6 transport.

    StaticPlanCompiler/topology work remains on CPU.  This adapter freezes its
    runtime rank-pair result into logical slots without embedding a transport
    object or a device pointer.
    """

    config.validate()
    peer_stride = 8 + config.max_tokens_per_peer * config.record_bytes
    peer_stride = (peer_stride + 7) & ~7
    descriptor_stride = config.world_size * peer_stride
    region_bytes = config.max_descriptors * descriptor_stride
    templates: list[CompiledRouteTemplate] = []
    table = [-1] * (config.world_size * config.world_size)
    capacities: list[int] = []
    legality: list[int] = []
    for src in range(config.world_size):
        for dst in range(config.world_size):
            route_id = len(templates)
            flags = int(ActionFlags.LOGICAL_OFFSETS)
            if src == dst:
                flags |= int(ActionFlags.SELF)
            templates.append(CompiledRouteTemplate(
                src_rank=src,
                dst_rank=dst,
                route_id=route_id,
                channel_id=dst,
                send_region_base=dst * peer_stride,
                recv_region_base=region_bytes + src * peer_stride,
                flags=flags,
            ))
            table[src * config.world_size + dst] = route_id
            capacities.append(config.max_tokens_per_peer)
            legality.append(1)
    return serialize_compiled_plan(
        config, tuple(templates), tuple(table), tuple(capacities),
        tuple(legality), descriptor_stride=descriptor_stride,
        region_bytes=region_bytes,
    )


def serialize_compiled_plan(
    config: SchedulerConfig,
    route_templates: tuple[CompiledRouteTemplate, ...],
    rank_pair_to_route: tuple[int, ...],
    capacity_table: tuple[int, ...],
    legality_flags: tuple[int, ...],
    *,
    descriptor_stride: int,
    region_bytes: int,
) -> CompiledPlanBlob:
    config.validate()
    route_count = len(route_templates)
    if len(rank_pair_to_route) != config.world_size * config.world_size:
        raise ValueError("rank-pair lookup shape mismatch")
    if len(capacity_table) != route_count or len(legality_flags) != route_count:
        raise ValueError("compiled route table shape mismatch")
    route_offset = _HEADER.size
    pair_offset = route_offset + route_count * _ROUTE.size
    capacity_offset = pair_offset + len(rank_pair_to_route) * _I32.size
    capacity_offset = (capacity_offset + 7) & ~7
    legality_offset = capacity_offset + route_count * _U64.size
    total_bytes = legality_offset + route_count * _U32.size
    output = bytearray(total_bytes)
    _HEADER.pack_into(
        output, 0, PLAN_MAGIC, PLAN_VERSION, _HEADER.size, total_bytes, 0,
        config.world_size, route_count, config.max_descriptors,
        config.max_chunks, config.record_bytes, config.max_tokens_per_peer,
        descriptor_stride, region_bytes, route_offset, pair_offset,
        capacity_offset, legality_offset,
    )
    cursor = route_offset
    for item in route_templates:
        _ROUTE.pack_into(
            output, cursor, item.src_rank, item.dst_rank, item.route_id,
            item.channel_id, item.send_region_base, item.recv_region_base,
            item.flags, 0,
        )
        cursor += _ROUTE.size
    cursor = pair_offset
    for value in rank_pair_to_route:
        _I32.pack_into(output, cursor, value); cursor += _I32.size
    cursor = capacity_offset
    for value in capacity_table:
        _U64.pack_into(output, cursor, value); cursor += _U64.size
    cursor = legality_offset
    for value in legality_flags:
        _U32.pack_into(output, cursor, value); cursor += _U32.size
    checksum = fnv1a64(bytes(output), zero_checksum_field=True)
    _U64.pack_into(output, _CHECKSUM_OFFSET, checksum)
    blob = CompiledPlanBlob(
        config=config, route_templates=route_templates,
        rank_pair_to_route=rank_pair_to_route,
        capacity_table=capacity_table, legality_flags=legality_flags,
        descriptor_stride=descriptor_stride, region_bytes=region_bytes,
        data=bytes(output), checksum=checksum,
    )
    validate_compiled_plan(blob)
    return blob


def validate_compiled_plan(plan: CompiledPlanBlob | bytes) -> bool:
    data = plan.data if isinstance(plan, CompiledPlanBlob) else bytes(plan)
    if len(data) < _HEADER.size:
        raise ValueError("compiled plan header truncated")
    values = _HEADER.unpack_from(data)
    (magic, version, header_bytes, total_bytes, checksum, world_size,
     route_count, max_descriptors, max_chunks, record_bytes,
     max_tokens_per_peer, descriptor_stride, region_bytes, route_offset,
     pair_offset, capacity_offset, legality_offset) = values
    if magic != PLAN_MAGIC or version != PLAN_VERSION or header_bytes != _HEADER.size:
        raise ValueError("compiled plan identity mismatch")
    if total_bytes != len(data):
        raise ValueError("compiled plan length mismatch")
    expected_capacity = (
        _HEADER.size + route_count * _ROUTE.size + world_size * world_size * 4 + 7
    ) & ~7
    expected_offsets = (
        _HEADER.size,
        _HEADER.size + route_count * _ROUTE.size,
        expected_capacity,
    )
    if (route_offset, pair_offset, capacity_offset) != expected_offsets:
        raise ValueError("compiled plan offset mismatch")
    if legality_offset + route_count * 4 != total_bytes:
        raise ValueError("compiled plan tail mismatch")
    if min(world_size, route_count, max_descriptors, max_chunks,
           record_bytes, max_tokens_per_peer, descriptor_stride, region_bytes) <= 0:
        raise ValueError("compiled plan contains invalid capacity")
    if fnv1a64(data, zero_checksum_field=True) != checksum:
        raise ValueError("compiled plan checksum mismatch")
    return True


__all__ = [
    "CompiledPlanBlob", "PLAN_MAGIC", "PLAN_VERSION",
    "compile_rank_pair_plan", "fnv1a64", "serialize_compiled_plan",
    "validate_compiled_plan",
]

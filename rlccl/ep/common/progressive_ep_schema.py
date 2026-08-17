"""Host-side mirrors for the R6-M7 commit boundary."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DescriptorCommit:
    commit_id: int
    descriptor_id: int
    chunk_id: int
    reveal_epoch: int
    token_begin: int
    token_count: int
    assignment_begin: int
    assignment_count: int
    authorized_dst_mask: int
    flags: int

    @classmethod
    def from_row(cls, row) -> "DescriptorCommit":
        return cls(*map(int, row))


@dataclass(frozen=True, slots=True)
class CommitPeerPlan:
    descriptor_id: int
    destination: int
    token_count: int
    route_id: int
    src_base_offset: int
    dst_base_offset: int
    flags: int

    @classmethod
    def from_row(cls, row) -> "CommitPeerPlan":
        return cls(*map(int, row))


__all__ = ["CommitPeerPlan", "DescriptorCommit"]

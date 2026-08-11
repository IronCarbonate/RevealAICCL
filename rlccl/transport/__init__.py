"""Reference transport substrates used by staged correctness gates."""

from .reference_a2av import (
    PAYLOAD_FIELDS,
    DestinationLayout,
    PackedShard,
    ProgressivePackingState,
    RouterAssignment,
    build_destination_layout,
    decode_records,
    pack_destination_layout,
    payload_checksum,
    payload_multiset_digest,
    verify_received_records,
)

__all__ = [
    "PAYLOAD_FIELDS",
    "DestinationLayout",
    "PackedShard",
    "ProgressivePackingState",
    "RouterAssignment",
    "build_destination_layout",
    "decode_records",
    "pack_destination_layout",
    "payload_checksum",
    "payload_multiset_digest",
    "verify_received_records",
]

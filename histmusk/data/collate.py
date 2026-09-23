"""Batch variable-length CellViT sets with an explicit validity mask."""

from __future__ import annotations

from typing import Any, Sequence

import torch


def multimodal_collate_fn(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    counts = torch.tensor([int(item["cell_count"]) for item in batch], dtype=torch.long)
    cell_dim = int(batch[0]["cell_embeddings"].shape[1])
    loaded_counts = [int(item["cell_embeddings"].shape[0]) for item in batch]
    max_cells = max(1, max(loaded_counts))
    cells = torch.zeros((len(batch), max_cells, cell_dim), dtype=torch.float32)
    mask = torch.zeros((len(batch), max_cells), dtype=torch.bool)
    for batch_index, item in enumerate(batch):
        count = loaded_counts[batch_index]
        if item.get("cell_loaded", True) and item["cell_embeddings"].shape != (int(counts[batch_index]), cell_dim):
            raise ValueError("cell_count does not match cell_embeddings shape")
        if count:
            cells[batch_index, :count].copy_(item["cell_embeddings"])
            mask[batch_index, :count] = True
    result = {
        "spot_index": torch.tensor([item["spot_index"] for item in batch], dtype=torch.long),
        "spot_id": [item["spot_id"] for item in batch],
        "sample_id": [item["sample_id"] for item in batch],
        "patient_id": [item["patient_id"] for item in batch],
        "slide_id": [item["slide_id"] for item in batch],
        "cell_embeddings": cells,
        "cell_mask": mask,
        "cell_count": counts,
        "conch_embedding": torch.stack([item["conch_embedding"] for item in batch]),
        "qwen_embedding": torch.stack([item["qwen_embedding"] for item in batch]),
        "coordinate": torch.stack([item["coordinate"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
    }
    has_base = ["base_prediction" in item for item in batch]
    if any(has_base) and not all(has_base):
        raise ValueError("Only part of the batch contains base predictions")
    if all(has_base):
        result["base_prediction"] = torch.stack(
            [item["base_prediction"] for item in batch]
        )
    return result

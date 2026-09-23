from __future__ import annotations

import torch

from histmusk.data.collate import multimodal_collate_fn
from histmusk.models.losses import SACTFLoss, build_loss


def _item(count: int, spot_index: int) -> dict:
    return {
        "spot_index": spot_index,
        "spot_id": str(spot_index),
        "sample_id": "sample",
        "patient_id": "patient",
        "slide_id": "slide",
        "cell_embeddings": torch.ones(count, 8),
        "cell_count": count,
        "conch_embedding": torch.ones(4),
        "qwen_embedding": torch.ones(3),
        "coordinate": torch.ones(2),
        "target": torch.ones(5),
    }


def test_collate_pads_variable_cell_sets() -> None:
    batch = multimodal_collate_fn([_item(0, 0), _item(3, 1)])
    assert batch["cell_embeddings"].shape == (2, 3, 8)
    assert batch["cell_mask"].dtype == torch.bool
    assert not batch["cell_mask"][0].any()
    assert batch["cell_mask"][1].all()
    assert batch["cell_count"].tolist() == [0, 3]


def test_loss_is_finite_for_constant_targets() -> None:
    prediction = torch.ones(4, 6, requires_grad=True)
    target = torch.ones(4, 6)
    outputs = {
        "prediction": prediction,
        "spot_prediction": prediction,
        "cell_gate": torch.zeros(4, 1),
        "text_gate": torch.zeros(4, 1),
        "cell_residual": torch.zeros(4, 8),
        "text_feature": torch.zeros(4, 8),
    }
    values = SACTFLoss()(outputs, target, prediction)
    assert all(torch.isfinite(value) for value in values.values())
    values["loss"].backward()
    assert torch.isfinite(prediction.grad).all()


def test_default_loss_profile() -> None:
    criterion = build_loss({})
    assert criterion.spot_pcc_weight == 0.1
    assert criterion.gene_pcc_weight == 0.1
    assert criterion.consistency_weight == 0.005

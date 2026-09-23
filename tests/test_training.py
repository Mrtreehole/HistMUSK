from __future__ import annotations

import torch

from histmusk.models import HistMUSKModel
from histmusk.models.losses import SACTFLoss
from histmusk.training.checkpointing import load_model_weights, save_checkpoint


def test_optimizer_step_and_checkpoint_roundtrip(tmp_path) -> None:
    model = HistMUSKModel(
        12,
        16,
        7,
        9,
        hidden_dim=32,
        num_heads=4,
        dropout=0.0,
        cell_dropout=0.0,
        text_dropout=0.0,
        text_fusion="concat",
    )
    batch = {
        "conch_embedding": torch.randn(4, 12),
        "qwen_embedding": torch.randn(4, 7),
        "cell_embeddings": torch.randn(4, 5, 16),
        "cell_mask": torch.ones(4, 5, dtype=torch.bool),
        "cell_count": torch.full((4,), 5),
        "target": torch.randn(4, 9),
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = next(model.parameters()).detach().clone()
    loss = SACTFLoss()(model(batch), batch["target"])["loss"]
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, next(model.parameters()).detach())

    checkpoint = tmp_path / "model.pt"
    save_checkpoint(
        checkpoint,
        model,
        optimizer,
        None,
        epoch=1,
        best_metric=0.0,
        config={},
        gene_names=[f"gene_{index}" for index in range(9)],
        target_scaler={},
    )
    clone = HistMUSKModel(
        12,
        16,
        7,
        9,
        hidden_dim=32,
        num_heads=4,
        dropout=0.0,
        cell_dropout=0.0,
        text_dropout=0.0,
        text_fusion="concat",
    )
    missing, unexpected = load_model_weights(clone, checkpoint, strict=True)
    assert not missing and not unexpected
    for source, restored in zip(model.parameters(), clone.parameters()):
        torch.testing.assert_close(source, restored)

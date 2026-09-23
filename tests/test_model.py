from __future__ import annotations

import torch

from histmusk.models import HistMUSKModel, build_model


def make_batch() -> dict[str, torch.Tensor]:
    return {
        "conch_embedding": torch.randn(3, 12),
        "qwen_embedding": torch.randn(3, 7),
        "cell_embeddings": torch.randn(3, 5, 16),
        "cell_mask": torch.tensor(
            [
                [False, False, False, False, False],
                [True, True, False, False, False],
                [True, True, True, True, True],
            ]
        ),
        "cell_count": torch.tensor([0, 2, 5]),
        "target": torch.randn(3, 9),
    }


def make_model(**kwargs) -> HistMUSKModel:
    return HistMUSKModel(
        conch_dim=12,
        cell_dim=16,
        qwen_dim=7,
        num_genes=9,
        hidden_dim=32,
        num_cell_prototypes=4,
        num_heads=4,
        dropout=0.0,
        **kwargs,
    )


def test_forward_is_empty_cell_safe() -> None:
    outputs = make_model().eval()(make_batch(), apply_modality_dropout=False)
    assert outputs["prediction"].shape == (3, 9)
    assert outputs["cell_attention"].shape == (3, 1, 4)
    assert torch.isfinite(outputs["prediction"]).all()
    assert outputs["cell_gate"][0].item() == 0.0
    assert torch.all((outputs["cell_gate"] >= 0) & (outputs["cell_gate"] <= 0.5))


def test_text_concat_is_active_and_receives_gradients() -> None:
    model = make_model(text_fusion="concat", text_dropout=0.0).train()
    batch = make_batch()
    full = model(batch, apply_modality_dropout=False)
    no_text = model(batch, modality_mode="no_text", apply_modality_dropout=False)
    assert torch.all(full["text_gate"] == 1)
    assert torch.all(no_text["text_gate"] == 0)
    assert not torch.equal(full["prediction"], no_text["prediction"])
    full["prediction"].square().mean().backward()
    gradient = model.text_encoder[1].weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_model_builder_validates_text_width() -> None:
    config = {
        "model": {
            "type": "histmusk",
            "expected_qwen_dim": 7,
            "hidden_dim": 32,
            "num_heads": 4,
            "text_fusion": "concat",
        }
    }
    dimensions = {"conch_dim": 12, "cell_dim": 16, "qwen_dim": 7, "num_genes": 9}
    assert build_model(config, dimensions)(make_batch())["prediction"].shape == (3, 9)

    dimensions["qwen_dim"] = 8
    try:
        build_model(config, dimensions)
    except ValueError as error:
        assert "expected 7, found 8" in str(error)
    else:
        raise AssertionError("A mismatched text embedding width must fail")

# Data format

HistMUSK aligns every modality by the explicit `spot_id` in `spots.parquet`.
Row-count equality is not accepted as evidence of alignment.

## Multimodal feature directory

```text
multimodal/
|-- features.zarr/
|-- spots.parquet
`-- manifest.json
```

`spots.parquet` must be sorted by a contiguous `spot_index` and contain:

- `spot_index`, `spot_id`, `sample_id`, `patient_id`, `slide_id`
- `cell_count`
- spatial/QC columns may be retained as additional metadata

`features.zarr` must contain:

| Array | Shape | Dtype | Meaning |
| --- | --- | --- | --- |
| `cell_embeddings` | `[M, D_cell]` | float32 | Cells sorted by spot |
| `cell_indptr` | `[N + 1]` | int64 | CSR boundaries for each spot |
| `conch_embeddings` | `[N, D_image]` | float32 | Image features; legacy key also used for UNI |
| `qwen_embeddings` | `[N, D_text]` | float32 | Text embeddings |
| `spatial_coordinates` | `[N, 2]` | float32 | Spot `(x, y)` coordinates |

Optional audit arrays include `cell_positions_wsi` and
`cell_to_spot_index`. The expected real-data dimensions are typically
`D_cell=1280`, `D_image=1024` for UNI, and `D_text=128` for the default model.

## Expression H5AD

The expression file must satisfy:

- `layers["counts"]`: non-negative integer counts.
- `X`: `log1p(count / selected_gene_library_size * 10000)`.
- unique genes in `var_names`.
- `obs_names`: `<sample_id>_count__<spot_id>`.

The suffix after the first `__` is joined to `spots.parquet::spot_id`. The
prefix, after removing `_count`, must equal `sample_id`. Training creates an
ID-aligned target cache and a train-only gene standardization scaler.

## Optional feature overrides

`--qwen-h5ad` accepts one embedding matrix in `obsm` and explicit IDs in
`obs["spot_id"]` or `obs_names`.

`--image-features-h5` accepts:

- `features`: `[N, D_image]` float matrix.
- `paths`: image paths whose filename stems are the `spot_id` values.

Both loaders reject duplicate, unknown, or silently missing IDs.

## Explicit split files

`--splits-file` overrides the default `target_dir/splits.parquet`. The table
must contain a unique `spot_id`, its target `spot_index`, and a `split` value
(`train`, `val`, `test`, or `excluded`) for every multimodal spot. Patient-level
LOPO uses one such immutable table per fold and passes a matching scaler through
`--target-scaler`; neither row-count equality nor implicit row order is used for
alignment.

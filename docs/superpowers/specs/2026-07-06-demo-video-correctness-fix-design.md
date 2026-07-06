# Demo Video Export Correctness Fix Design

## Goal

Make `scripts/export_demo_video.py` operate correctly with the repository's
current memmap metadata and deterministic split manifests, while preserving the
existing inference, rendering, MP4, and GIF behavior.

## CLI Contract

- `--action` accepts the exact action label stored in `meta.npz`, such as
  `A01`, rather than an integer ordinal.
- `--subject` accepts an exact subject label such as `S03`. When omitted, the
  exporter selects the lexicographically first subject available for the
  requested action and prints the available subjects and selected value.
- `--split-manifest` and `--manifest-key` are optional but must be supplied
  together. Without them, the exporter uses the complete memmap dataset.
- `--gif-only` and `--mp4-only` are mutually exclusive.
- FPS must be greater than zero. Custom width and height must be positive even
  integers so H.264 `yuv420p` encoding is valid.

## Data and Model Flow

When no manifest is supplied, the exporter constructs `MemmapDataset` with the
existing global-minmax data and all indices. When a manifest is supplied, it
loads the manifest relative to the dataset root and builds the dataset from the
requested indices with `source_train_normalization`.

Checkpoint loading receives the manifest hash when a manifest is active. This
rejects a checkpoint produced from a different deterministic split instead of
silently exporting incomparable predictions.

Frame collection matches exact action and subject metadata. It reports clear
errors containing the available values when either label is unavailable. Frame
ordering continues to use the dataset's deterministic absolute frame indices.
The CSI conversion remains `[64, 3, 114] -> [1, 3, 114, 64]`, matching
`memmap_collate_fn`.

## Error Handling

Argument-level conflicts and scalar validation are handled by argparse so the
process exits with status 2 and a concise usage error. Dataset label failures
raise `ValueError` with available labels. Existing ffmpeg availability and
subprocess error handling remains unchanged.

## Tests

Tests will be added before production changes and observed failing for the
missing behavior. Focused coverage will verify:

- exact `A01` action and `Sxx` subject selection;
- deterministic default subject selection;
- unavailable action and subject diagnostics;
- paired manifest arguments and manifest-backed dataset normalization;
- checkpoint manifest-hash propagation;
- mutually exclusive output flags;
- rejection of non-positive FPS and invalid video dimensions;
- preservation of existing MPJPE and PNG rendering behavior.

After the focused red-green cycle, verification will run the focused test file,
the complete pytest suite, Python syntax compilation, and Git whitespace checks.

## Scope

The change is limited to the demo-video script and its focused tests. It does
not refactor evaluation modules, alter model architecture, modify datasets, or
change generated-output formats.

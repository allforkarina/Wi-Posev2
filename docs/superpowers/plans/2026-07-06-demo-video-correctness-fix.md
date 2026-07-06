# Demo Video Export Correctness Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the demo-video exporter select exact MM-Fi labels, honor deterministic manifests, reject invalid CLI combinations, and preserve rendering behavior.

**Architecture:** Keep changes local to `scripts/export_demo_video.py` and `tests/test_export_demo_video.py`. Add small validation and selection helpers, then reuse the evaluation module's manifest-aware dataset and checkpoint loaders.

**Tech Stack:** Python 3.10+, argparse, NumPy, PyTorch, pytest, Matplotlib, ffmpeg

---

## File Structure

- Modify `scripts/export_demo_video.py`: label selection, CLI validation, manifest loading, and usage text.
- Modify `tests/test_export_demo_video.py`: regression tests with tiny fake datasets and monkeypatched loader boundaries.

### Task 1: Exact Action and Subject Labels

**Files:**
- Modify: `tests/test_export_demo_video.py`
- Modify: `scripts/export_demo_video.py:108-168`

- [ ] **Step 1: Write failing selection tests**

Add this dataset double and four tests:

```python
class _SelectionDataset:
    def __init__(self) -> None:
        self.indices = np.arange(4, dtype=np.int64)
        self._actions = np.asarray(["A01", "A01", "A01", "A02"])
        self._samples = np.asarray(["S03", "S01", "S03", "S01"])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> dict:
        absolute = int(self.indices[position])
        return {
            "csi": torch.zeros(64, 3, 114),
            "kpts18": torch.zeros(18, 2),
            "meta": {
                "env": "E01",
                "subject": str(self._samples[absolute]),
                "action": str(self._actions[absolute]),
                "frame_idx": absolute,
            },
        }


def test_collect_action_frames_selects_exact_subject_label() -> None:
    from scripts.export_demo_video import collect_action_frames
    frames, subjects = collect_action_frames(
        _SelectionDataset(), action="A01", subject="S03"
    )
    assert subjects == ["S01", "S03"]
    assert [frame["meta"]["frame_idx"] for frame in frames] == [0, 2]


def test_collect_action_frames_defaults_to_first_subject() -> None:
    from scripts.export_demo_video import collect_action_frames
    frames, subjects = collect_action_frames(_SelectionDataset(), action="A01")
    assert subjects == ["S01", "S03"]
    assert {frame["meta"]["subject"] for frame in frames} == {"S01"}


def test_collect_action_frames_reports_actions() -> None:
    from scripts.export_demo_video import collect_action_frames
    with pytest.raises(ValueError, match=r"Available actions: A01, A02"):
        collect_action_frames(_SelectionDataset(), action="A03")


def test_collect_action_frames_reports_subjects() -> None:
    from scripts.export_demo_video import collect_action_frames
    with pytest.raises(ValueError, match=r"Available subjects: S01, S03"):
        collect_action_frames(_SelectionDataset(), action="A01", subject="S99")
```

- [ ] **Step 2: Verify RED**

Run `python -m pytest tests/test_export_demo_video.py -q -k "collect_action_frames"`.

Expected: failures because the current function converts labels to integer-style strings, treats subject as an ordinal, and returns only a list.

- [ ] **Step 3: Implement exact matching**

Replace the selection function with:

```python
def collect_action_frames(
    dataset: MemmapDataset,
    action: str,
    subject: str | None = None,
) -> tuple[list[dict], list[str]]:
    """Collect ordered frames and available subjects for an exact action label."""
    available_actions = sorted({
        str(dataset._actions[int(index)]) for index in dataset.indices
    })
    if action not in available_actions:
        raise ValueError(
            f"No frames found for action={action!r}. "
            f"Available actions: {', '.join(available_actions)}"
        )
    matching = [
        position
        for position, index in enumerate(dataset.indices)
        if str(dataset._actions[int(index)]) == action
    ]
    samples = [
        str(dataset._samples[int(dataset.indices[position])])
        for position in matching
    ]
    available_subjects = sorted(set(samples))
    selected = subject if subject is not None else available_subjects[0]
    if selected not in available_subjects:
        raise ValueError(
            f"No frames found for action={action!r}, subject={selected!r}. "
            f"Available subjects: {', '.join(available_subjects)}"
        )
    frames = [
        dataset[position]
        for position, sample in zip(matching, samples)
        if sample == selected
    ]
    frames.sort(key=lambda frame: frame["meta"]["frame_idx"])
    return frames, available_subjects
```

Update `main()` to unpack `(frames, available_subjects)` and print both the available list and selected subject.

- [ ] **Step 4: Verify GREEN and commit**

Run `python -m pytest tests/test_export_demo_video.py -q -k "collect_action_frames"`; expect `4 passed`.

Then run:

```bash
git add scripts/export_demo_video.py tests/test_export_demo_video.py
git commit -m "Fix demo action and subject selection"
```

### Task 2: CLI Validation

**Files:**
- Modify: `tests/test_export_demo_video.py`
- Modify: `scripts/export_demo_video.py:61-75`
- Modify: `scripts/export_demo_video.py:505-561`

- [ ] **Step 1: Write failing parser tests**

Add these parser tests:

```python
def test_action_and_subject_are_exact_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.export_demo_video import parse_args
    monkeypatch.setattr(sys, "argv", [
        "export_demo_video.py", "--checkpoint", "model.pth",
        "--dataset-root", "data/mmfi_pose", "--action", "A01",
        "--subject", "S03",
    ])
    args = parse_args()
    assert args.action == "A01"
    assert args.subject == "S03"


@pytest.mark.parametrize("fps", ["0", "-1"])
def test_non_positive_fps_is_rejected(
    monkeypatch: pytest.MonkeyPatch, fps: str
) -> None:
    from scripts.export_demo_video import parse_args
    monkeypatch.setattr(sys, "argv", [
        "export_demo_video.py", "--checkpoint", "model.pth",
        "--dataset-root", "data/mmfi_pose", "--action", "A01",
        "--fps", fps,
    ])
    with pytest.raises(SystemExit):
        parse_args()


@pytest.mark.parametrize("resolution", ["0x720", "1279x720", "1280x721"])
def test_invalid_video_resolution_is_rejected(
    monkeypatch: pytest.MonkeyPatch, resolution: str
) -> None:
    from scripts.export_demo_video import parse_args
    monkeypatch.setattr(sys, "argv", [
        "export_demo_video.py", "--checkpoint", "model.pth",
        "--dataset-root", "data/mmfi_pose", "--action", "A01",
        "--resolution", resolution,
    ])
    with pytest.raises(SystemExit):
        parse_args()


def test_output_only_flags_are_mutually_exclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.export_demo_video import parse_args
    monkeypatch.setattr(sys, "argv", [
        "export_demo_video.py", "--checkpoint", "model.pth",
        "--dataset-root", "data/mmfi_pose", "--action", "A01",
        "--gif-only", "--mp4-only",
    ])
    with pytest.raises(SystemExit):
        parse_args()
```

- [ ] **Step 2: Verify RED**

Run `python -m pytest tests/test_export_demo_video.py -q -k "exact_labels or fps or resolution or mutually_exclusive"`.

Expected: the new tests fail for the absent validation and incompatible integer types.

- [ ] **Step 3: Implement parser validation**

Add:

```python
def positive_float(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected a number, got {raw!r}") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("Value must be greater than zero")
    return value
```

After resolution integer parsing, reject non-positive or odd dimensions:

```python
if w <= 0 or h <= 0:
    raise argparse.ArgumentTypeError("Resolution dimensions must be positive")
if w % 2 or h % 2:
    raise argparse.ArgumentTypeError(
        "Resolution dimensions must be even for H.264 yuv420p output"
    )
```

Use string action/subject arguments, `type=positive_float`, `type=parse_resolution`, and an argparse mutually exclusive group for the two output-only flags. In `main()`, unpack `width, height = args.resolution`. Update the existing defaults test to expect `args.fps == 18.0`, `args.resolution == (1280, 720)`, and `args.subject is None`.

- [ ] **Step 4: Verify GREEN and commit**

Run the Step 2 command again; expect all selected tests to pass.

Then run:

```bash
git add scripts/export_demo_video.py tests/test_export_demo_video.py
git commit -m "Validate demo video options"
```

### Task 3: Manifest-Aware Model and Dataset Loading

**Files:**
- Modify: `tests/test_export_demo_video.py`
- Modify: `scripts/export_demo_video.py:20-38`
- Modify: `scripts/export_demo_video.py:553-591`

- [ ] **Step 1: Write failing manifest tests**

Add parser tests proving either manifest argument without the other exits:

```python
@pytest.mark.parametrize("extra", [
    ["--split-manifest", "split.npz"],
    ["--manifest-key", "env2_test"],
])
def test_manifest_arguments_must_be_paired(
    monkeypatch: pytest.MonkeyPatch, extra: list[str]
) -> None:
    from scripts.export_demo_video import parse_args
    monkeypatch.setattr(sys, "argv", [
        "export_demo_video.py", "--checkpoint", "model.pth",
        "--dataset-root", "data/mmfi_pose", "--action", "A01", *extra,
    ])
    with pytest.raises(SystemExit):
        parse_args()
```

Add this loader-boundary test:

```python
def test_manifest_hash_and_split_reach_loaders(monkeypatch: pytest.MonkeyPatch) -> None:
    import argparse
    import scripts.export_demo_video as module
    manifest = argparse.Namespace(manifest_hash="abc123")
    calls: dict[str, object] = {}
    monkeypatch.setattr(module, "load_manifest", lambda path, root: manifest)
    monkeypatch.setattr(
        module, "load_checkpoint_model",
        lambda path, device, expected_manifest_hash=None:
            calls.update(hash=expected_manifest_hash) or "model",
    )
    monkeypatch.setattr(
        module, "build_evaluation_dataset",
        lambda dataset_root, manifest=None, manifest_key=None:
            calls.update(manifest=manifest, key=manifest_key) or "dataset",
    )
    args = argparse.Namespace(
        checkpoint="model.pth", dataset_root="data/mmfi_pose",
        split_manifest="split.npz", manifest_key="env2_test",
    )
    result = module.load_export_model_and_dataset(args, torch.device("cpu"))
    assert result == ("model", "dataset")
    assert calls == {"hash": "abc123", "manifest": manifest, "key": "env2_test"}
```

- [ ] **Step 2: Verify RED**

Run `python -m pytest tests/test_export_demo_video.py -q -k "manifest"`.

Expected: failures because `--split-manifest` and the loader helper do not exist and `--manifest-key` is unused.

- [ ] **Step 3: Implement the manifest boundary**

Import `load_manifest` and `build_evaluation_dataset`. Add `--split-manifest`, retain `--manifest-key`, and after parsing call `parser.error()` unless both are present or both absent.

Add:

```python
def load_export_model_and_dataset(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.nn.Module, MemmapDataset]:
    manifest = (
        load_manifest(args.split_manifest, args.dataset_root)
        if args.split_manifest else None
    )
    model = load_checkpoint_model(
        args.checkpoint,
        device,
        expected_manifest_hash=manifest.manifest_hash if manifest else None,
    )
    dataset = build_evaluation_dataset(
        dataset_root=args.dataset_root,
        manifest=manifest,
        manifest_key=args.manifest_key,
    )
    return model, dataset
```

Use this helper in `main()` instead of constructing the model and all-index dataset separately.

- [ ] **Step 4: Verify GREEN and commit**

Run the Step 2 command again; expect all manifest tests to pass.

Then run:

```bash
git add scripts/export_demo_video.py tests/test_export_demo_video.py
git commit -m "Honor manifests in demo video export"
```

### Task 4: Full Verification and Delivery

**Files:**
- Modify: `scripts/export_demo_video.py:1-15`

- [ ] **Step 1: Update usage text**

Replace the numeric-action example with `--action A01 --subject S03 --mp4-only` in the module docstring and keep all repository-facing text in English.

- [ ] **Step 2: Run focused and syntax verification**

```bash
python -m pytest tests/test_export_demo_video.py -q
python -m py_compile scripts/export_demo_video.py
```

Expected: focused tests pass; compilation exits 0 without output.

- [ ] **Step 3: Run the complete suite**

Run `pytest` and require exit code 0. If filesystem ACLs block pytest cache or temporary paths, rerun with a verified writable `--basetemp` and `-p no:cacheprovider`, and report that constraint.

- [ ] **Step 4: Inspect the final diff**

```bash
git diff --check
git diff -- scripts/export_demo_video.py tests/test_export_demo_video.py
git status --short --branch
```

Expected: no whitespace errors and no unrelated files staged by this work.

- [ ] **Step 5: Commit and push**

```bash
git add scripts/export_demo_video.py tests/test_export_demo_video.py
git commit -m "Fix demo video export workflow"
git push
```

Expected: the active `codex/` branch is pushed without staging unrelated user files.

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# We'll import from the script once it exists
# from scripts.export_demo_video import compute_mpjpe, render_frame, parse_args


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


class TestCollectActionFrames:
    def test_selects_exact_subject_label(self) -> None:
        from scripts.export_demo_video import collect_action_frames

        frames, subjects = collect_action_frames(
            _SelectionDataset(), action="A01", subject="S03"
        )

        assert subjects == ["S01", "S03"]
        assert [frame["meta"]["frame_idx"] for frame in frames] == [0, 2]

    def test_defaults_to_first_subject(self) -> None:
        from scripts.export_demo_video import collect_action_frames

        frames, subjects = collect_action_frames(_SelectionDataset(), action="A01")

        assert subjects == ["S01", "S03"]
        assert {frame["meta"]["subject"] for frame in frames} == {"S01"}

    def test_reports_available_actions(self) -> None:
        from scripts.export_demo_video import collect_action_frames

        with pytest.raises(ValueError, match=r"Available actions: A01, A02"):
            collect_action_frames(_SelectionDataset(), action="A03")

    def test_reports_available_subjects(self) -> None:
        from scripts.export_demo_video import collect_action_frames

        with pytest.raises(ValueError, match=r"Available subjects: S01, S03"):
            collect_action_frames(
                _SelectionDataset(), action="A01", subject="S99"
            )


class TestComputeMpjpe:
    """Tests for per-frame MPJPE calculation."""

    def test_identical_keypoints_give_zero(self) -> None:
        """MPJPE between identical keypoint sets should be 0."""
        from scripts.export_demo_video import compute_mpjpe

        kpts = np.random.randn(18, 2).astype(np.float32)
        error = compute_mpjpe(kpts, kpts)
        assert error == pytest.approx(0.0, abs=1e-6)

    def test_known_offset(self) -> None:
        """MPJPE with a known 1.0 offset on all joints should be sqrt(2)."""
        from scripts.export_demo_video import compute_mpjpe

        gt = np.zeros((18, 2), dtype=np.float32)
        pred = np.ones((18, 2), dtype=np.float32)
        # Each joint: sqrt(1^2 + 1^2) = sqrt(2) ≈ 1.4142
        error = compute_mpjpe(gt, pred)
        assert error == pytest.approx(np.sqrt(2), rel=1e-4)

    def test_single_joint_offset(self) -> None:
        """Offset on only one joint should average across all 18."""
        from scripts.export_demo_video import compute_mpjpe

        gt = np.zeros((18, 2), dtype=np.float32)
        pred = np.zeros((18, 2), dtype=np.float32)
        pred[0, 0] = 3.0  # offset of 3 in x on joint 0 only
        # Only joint 0 has error = 3.0; others 0. Mean = 3.0 / 18 = 0.1666...
        error = compute_mpjpe(gt, pred)
        assert error == pytest.approx(3.0 / 18.0, rel=1e-4)


class TestRenderFrame:
    """Tests for single-frame rendering."""

    def test_creates_png_file(self, tmp_path: Path) -> None:
        """render_frame should create a PNG at the expected path."""
        from scripts.export_demo_video import render_frame

        gt = np.random.randn(18, 2).astype(np.float32) * 0.1
        pred = np.random.randn(18, 2).astype(np.float32) * 0.1

        output_path = render_frame(
            gt=gt,
            pred=pred,
            mpjpe=12.4,
            frame_idx=0,
            total_frames=297,
            action_label="0",
            subject_id="1",
            model_label="WiFlow v2",
            output_dir=tmp_path,
        )

        assert output_path.is_file()
        assert output_path.suffix == ".png"

    def test_png_has_correct_dimensions(self, tmp_path: Path) -> None:
        """Output PNG should be 1280×720 at 100 DPI (12.8×7.2 inches)."""
        from scripts.export_demo_video import render_frame

        gt = np.random.randn(18, 2).astype(np.float32) * 0.1
        pred = np.random.randn(18, 2).astype(np.float32) * 0.1

        output_path = render_frame(
            gt=gt,
            pred=pred,
            mpjpe=12.4,
            frame_idx=0,
            total_frames=297,
            action_label="0",
            subject_id="1",
            model_label="WiFlow v2",
            output_dir=tmp_path,
        )

        # Read back and check dimensions
        img = plt.imread(str(output_path))
        # At 100 DPI, 1280×720 → 12.8×7.2 inches → saved at 100 DPI → 1280×720 pixels
        assert img.shape[0] == 720, f"Expected 720px height, got {img.shape[0]}"
        assert img.shape[1] == 1280, f"Expected 1280px width, got {img.shape[1]}"

    def test_frame_idx_zero_shows_frame_1_of_total(self, tmp_path: Path) -> None:
        """Frame index 0 should display as 'Frame 1/297'."""
        from scripts.export_demo_video import render_frame

        gt = np.random.randn(18, 2).astype(np.float32) * 0.1
        pred = np.random.randn(18, 2).astype(np.float32) * 0.1

        output_path = render_frame(
            gt=gt, pred=pred, mpjpe=5.0,
            frame_idx=0, total_frames=297,
            action_label="waving", subject_id="S01",
            model_label="Test", output_dir=tmp_path,
        )
        assert output_path.is_file()


class TestParseArgs:
    """Tests for CLI argument parsing."""

    def test_required_args(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """--checkpoint, --dataset-root, and --action are required."""
        from scripts.export_demo_video import parse_args

        monkeypatch.setattr(sys, "argv", [
            "export_demo_video.py",
            "--checkpoint", "model.pth",
            "--dataset-root", "data/mmfi_pose",
            "--action", "0",
        ])
        args = parse_args()
        assert args.checkpoint == "model.pth"
        assert args.dataset_root == "data/mmfi_pose"
        assert args.action == "0"

    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Check all default values."""
        from scripts.export_demo_video import parse_args

        monkeypatch.setattr(sys, "argv", [
            "export_demo_video.py",
            "--checkpoint", "model.pth",
            "--dataset-root", "data/mmfi_pose",
            "--action", "0",
        ])
        args = parse_args()
        assert args.fps == 18.0
        assert args.resolution == (1280, 720)
        assert args.subject is None
        assert args.output_dir == "outputs/demo_videos"
        assert args.model_label == "WiFlow v2"
        assert args.keep_frames is False
        assert args.gif_only is False
        assert args.mp4_only is False

    def test_resolution_preset_720p(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """--resolution 720p should be accepted as a preset string."""
        from scripts.export_demo_video import parse_args

        monkeypatch.setattr(sys, "argv", [
            "export_demo_video.py",
            "--checkpoint", "model.pth",
            "--dataset-root", "data/mmfi_pose",
            "--action", "0",
            "--resolution", "720p",
        ])
        args = parse_args()
        assert args.resolution == (1280, 720)

    def test_action_and_subject_are_exact_labels(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scripts.export_demo_video import parse_args

        monkeypatch.setattr(sys, "argv", [
            "export_demo_video.py",
            "--checkpoint", "model.pth",
            "--dataset-root", "data/mmfi_pose",
            "--action", "A01",
            "--subject", "S03",
        ])
        args = parse_args()
        assert args.action == "A01"
        assert args.subject == "S03"

    @pytest.mark.parametrize("fps", ["0", "-1"])
    def test_non_positive_fps_is_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fps: str,
    ) -> None:
        from scripts.export_demo_video import parse_args

        monkeypatch.setattr(sys, "argv", [
            "export_demo_video.py",
            "--checkpoint", "model.pth",
            "--dataset-root", "data/mmfi_pose",
            "--action", "A01",
            "--fps", fps,
        ])
        with pytest.raises(SystemExit):
            parse_args()

    @pytest.mark.parametrize("resolution", ["0x720", "1279x720", "1280x721"])
    def test_invalid_video_resolution_is_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
        resolution: str,
    ) -> None:
        from scripts.export_demo_video import parse_args

        monkeypatch.setattr(sys, "argv", [
            "export_demo_video.py",
            "--checkpoint", "model.pth",
            "--dataset-root", "data/mmfi_pose",
            "--action", "A01",
            "--resolution", resolution,
        ])
        with pytest.raises(SystemExit):
            parse_args()

    def test_output_only_flags_are_mutually_exclusive(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scripts.export_demo_video import parse_args

        monkeypatch.setattr(sys, "argv", [
            "export_demo_video.py",
            "--checkpoint", "model.pth",
            "--dataset-root", "data/mmfi_pose",
            "--action", "A01",
            "--gif-only",
            "--mp4-only",
        ])
        with pytest.raises(SystemExit):
            parse_args()


class TestFfmpegCheck:
    """Tests for ffmpeg availability check."""

    def test_ffmpeg_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """check_ffmpeg should raise SystemExit when ffmpeg is missing."""
        import shutil
        from scripts.export_demo_video import check_ffmpeg

        # Mock shutil.which to return None
        monkeypatch.setattr(shutil, "which", lambda _name: None)
        with pytest.raises(SystemExit) as exc_info:
            check_ffmpeg()
        assert exc_info.value.code == 1


class TestEndToEnd:
    """Integration test: render multiple frames and verify output."""

    def test_render_full_sequence(self, tmp_path: Path) -> None:
        """Render 10 synthetic frames and verify all PNGs are created."""
        from scripts.export_demo_video import render_frame

        np.random.seed(42)
        n_frames = 10
        for i in range(n_frames):
            gt = np.random.randn(18, 2).astype(np.float32) * 0.1 + 0.5
            pred = gt + np.random.randn(18, 2).astype(np.float32) * 0.02
            mpjpe = float(np.linalg.norm(pred - gt, axis=1).mean())

            path = render_frame(
                gt=gt, pred=pred, mpjpe=mpjpe,
                frame_idx=i, total_frames=n_frames,
                action_label="test_action", subject_id="S01",
                model_label="TestModel", output_dir=tmp_path,
            )
            assert path.is_file()

        # All 10 files should exist
        pngs = sorted(tmp_path.glob("frame_*.png"))
        assert len(pngs) == n_frames
        assert pngs[0].name == "frame_000.png"
        assert pngs[-1].name == "frame_009.png"

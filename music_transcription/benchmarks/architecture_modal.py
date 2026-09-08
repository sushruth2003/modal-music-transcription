"""Frozen-checkpoint architecture comparisons; isolated from the production app."""

import hashlib
import io
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import modal

app = modal.App("music-transcription-architectures")
REVISION = "5e66c1ea173a8186e0d20432b841d3180cc015b5"
CHECKPOINTS = {
    "yourmt3_t5": "notask_all_cross_v6_xk2_amp0811_gm_ext_plus_nops_b72/model.ckpt",
    "yourmt3_moe": "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops/last.ckpt",
}


def fetch_yourmt3():
    from huggingface_hub import snapshot_download

    snapshot_download(
        "mimbres/YourMT3",
        repo_type="space",
        revision=REVISION,
        local_dir="/opt/yourmt3",
        allow_patterns=["amt/src/*", "model_helper.py"]
        + [
            f"amt/logs/2024/{p.rsplit('/', 1)[0]}/checkpoints/{p.rsplit('/', 1)[1]}"
            for p in CHECKPOINTS.values()
        ],
    )


base = modal.Image.debian_slim(python_version="3.11").apt_install("ffmpeg", "libsndfile1")
your_image = (
    base.uv_pip_install(
        "torch==2.5.1",
        "torchaudio==2.5.1",
        "numpy==1.26.4",
        "transformers==4.45.1",
        "lightning==2.4.0",
        "librosa==0.10.2.post1",
        "einops==0.8.0",
        "wandb==0.18.7",
        "deprecated==1.2.15",
        "mir-eval==0.8.2",
        "mido==1.3.3",
        "soundfile==0.13.1",
        "huggingface-hub==0.25.2",
        "pretty-midi==0.2.11",
    )
    .run_function(fetch_yourmt3)
    .env({"WANDB_MODE": "disabled", "HF_HUB_DISABLE_TELEMETRY": "1"})
)
transkun_image = base.uv_pip_install(
    "torch==2.5.1",
    "torchaudio==2.5.1",
    "transkun==2.0.1",
    "numpy==1.26.4",
    "soundfile==0.13.1",
    "setuptools<81",
)
basic_image = (
    base.uv_pip_install("basic-pitch==0.4.0", extra_options="--no-deps")
    .uv_pip_install(
        "onnxruntime==1.20.1",
        "numpy==1.26.4",
        "librosa==0.10.2.post1",
        "mir-eval==0.8.2",
        "pretty-midi==0.2.11",
        "resampy==0.4.2",
        "scikit-learn==1.5.2",
        "scipy==1.14.1",
        "soundfile==0.13.1",
    )
    .env({"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"})
)


def metadata(checkpoint):
    import importlib.metadata

    return {
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
    }


def validate_audio(data):
    import soundfile as sf

    audio, sr = sf.read(io.BytesIO(data), dtype="float32")
    if sr != 16000 or audio.ndim != 1 or not 0 < len(audio) <= 480000:
        raise ValueError("Expected the frozen <=30 second mono 16 kHz excerpts")
    return audio, sr


@app.cls(
    image=your_image,
    gpu="L4",
    cpu=2,
    memory=12288,
    max_containers=1,
    scaledown_window=30,
    timeout=900,
)
class YourMT3:
    architecture: str = modal.parameter()

    @modal.enter()
    def load(self):
        import os
        import sys

        import torch

        os.chdir("/opt/yourmt3")
        sys.path[:0] = ["/opt/yourmt3", "/opt/yourmt3/amt/src"]
        from model_helper import load_model_checkpoint

        torch.set_num_threads(2)
        torch.manual_seed(20260905)
        exp, name = CHECKPOINTS[self.architecture].rsplit("/", 1)
        args = [f"{exp}@{name}", "-p", "2024", "-pr", "32"]
        if self.architecture == "yourmt3_moe":
            args += [
                "-tk",
                "mc13_full_plus_256",
                "-dec",
                "multi-t5",
                "-nl",
                "26",
                "-enc",
                "perceiver-tf",
                "-sqr",
                "1",
                "-ff",
                "moe",
                "-wf",
                "4",
                "-nmoe",
                "8",
                "-kmoe",
                "2",
                "-act",
                "silu",
                "-epe",
                "rope",
                "-rp",
                "1",
                "-ac",
                "spec",
                "-hop",
                "300",
                "-atc",
                "1",
            ]
        started = time.perf_counter()
        self.model = load_model_checkpoint(args, device="cpu").to("cuda").eval()
        torch.cuda.synchronize()
        self.info = metadata(f"amt/logs/2024/{exp}/checkpoints/{name}")
        self.info.update(
            revision=REVISION,
            args=args,
            load_seconds=time.perf_counter() - started,
            precision="float32",
            parameters=sum(p.numel() for p in self.model.parameters()),
        )

    @modal.method()
    def transcribe(self, data: bytes):
        import model_helper
        import torch
        from config.vocabulary import MT3_FULL_PLUS
        from utils.utils import create_inverse_vocab, write_model_output_as_midi

        # The author's demo intentionally collapses solo strings into an ensemble patch.
        # Preserve the model's predicted MT3 labels for transcription, retaining both exports.
        detailed_vocab = create_inverse_vocab(MT3_FULL_PLUS)

        def export_both(notes, output_dir, track_name, output_inverse_vocab):
            write_model_output_as_midi(notes, output_dir, track_name, output_inverse_vocab)
            write_model_output_as_midi(notes, output_dir, track_name + ".detailed", detailed_vocab)

        validate_audio(data)
        Path("/tmp/excerpt.wav").write_bytes(data)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        original_writer = model_helper.write_model_output_as_midi
        model_helper.write_model_output_as_midi = export_both
        try:
            with torch.inference_mode():
                output = model_helper.transcribe(
                    self.model, {"filepath": "/tmp/excerpt.wav", "track_name": "excerpt"}
                )
        finally:
            model_helper.write_model_output_as_midi = original_writer
        torch.cuda.synchronize()
        return {
            "midi": Path("model_output/excerpt.detailed.mid").read_bytes(),
            "demo_midi": Path(output).read_bytes(),
            "export_policy": "MT3_FULL_PLUS predicted labels; demo's gm_ext_plus export retained separately",
            "seconds": time.perf_counter() - started,
            "peak_memory_bytes": torch.cuda.max_memory_allocated(),
            "device": torch.cuda.get_device_name(0),
            "model": self.info,
        }


@app.cls(
    image=transkun_image,
    gpu="L4",
    cpu=2,
    memory=8192,
    max_containers=1,
    scaledown_window=30,
    timeout=900,
)
class TransKun:
    @modal.enter()
    def load(self):
        import moduleconf
        import torch
        import transkun

        torch.set_num_threads(2)
        root = Path(transkun.__file__).parent / "pretrained"
        conf = moduleconf.parseFromFile(str(root / "2.0.conf"))["Model"]
        started = time.perf_counter()
        self.model = conf.module.TransKun(conf=conf.config).to("cuda")
        ckpt = torch.load(root / "2.0.pt", map_location="cuda", weights_only=False)
        self.model.load_state_dict(ckpt.get("best_state_dict", ckpt.get("state_dict")), strict=True)
        self.model.eval()
        torch.cuda.synchronize()
        self.info = metadata(root / "2.0.pt")
        self.info.update(
            config=(root / "2.0.conf").read_text(),
            load_seconds=time.perf_counter() - started,
            precision="float32",
            parameters=sum(p.numel() for p in self.model.parameters()),
        )

    @modal.method()
    def transcribe(self, data: bytes):
        import soxr
        import torch
        from transkun.Data import writeMidi

        audio, sr = validate_audio(data)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        x = torch.from_numpy(soxr.resample(audio[:, None], sr, self.model.fs)).to("cuda")
        with torch.inference_mode():
            notes = self.model.transcribe(
                x, stepInSecond=None, segmentSizeInSecond=None, discardSecondHalf=False
            )
        output = io.BytesIO()
        writeMidi(notes).write(output)
        torch.cuda.synchronize()
        return {
            "midi": output.getvalue(),
            "seconds": time.perf_counter() - started,
            "peak_memory_bytes": torch.cuda.max_memory_allocated(),
            "device": torch.cuda.get_device_name(0),
            "model": self.info,
        }


@app.cls(image=basic_image, cpu=2, memory=4096, max_containers=1, scaledown_window=30, timeout=900)
class BasicPitch:
    @modal.enter()
    def load(self):
        import basic_pitch
        import onnxruntime as ort
        from basic_pitch.inference import Model

        path = next(Path(basic_pitch.__file__).parent.rglob("*.onnx"))
        started = time.perf_counter()
        self.model = Model(path)
        # Avoid ONNX allocating threads for every host core outside the 2-CPU allocation.
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.model.model = ort.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"]
        )
        self.info = metadata(path)
        self.info.update(
            load_seconds=time.perf_counter() - started, precision="float32", backend="ONNX CPU"
        )

    @modal.method()
    def transcribe(self, data: bytes):
        from basic_pitch.inference import predict

        validate_audio(data)
        path = Path("/tmp/excerpt.wav")
        path.write_bytes(data)
        started = time.perf_counter()
        _, midi, _ = predict(path, self.model)
        output = io.BytesIO()
        midi.write(output)
        return {
            "midi": output.getvalue(),
            "seconds": time.perf_counter() - started,
            "device": "2 CPU cores",
            "model": self.info,
        }


@app.local_entrypoint()
def main(
    model: str,
    execute: bool = False,
    limit: int = 32,
    output: str = "outputs/quality-architectures-20260906",
):
    if model not in {*CHECKPOINTS, "transkun", "basic_pitch"} or not 1 <= limit <= 32:
        raise ValueError("Unknown model or invalid suite limit")
    manifest = Path("data/quality-suite/manifest.json")
    data = json.loads(manifest.read_text())
    examples = [e for e in data["examples"] if model != "transkun" or e["dataset"] == "MAESTRO"][
        :limit
    ]
    if not examples:
        raise ValueError("Empty dataset selection")
    print(f"{model}: {len(examples)} frozen excerpts; author defaults, no reference conditioning")
    if not execute:
        return
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    frozen = root / "manifest.json"
    if frozen.exists() and json.loads(frozen.read_text()) != data:
        raise ValueError("Cannot resume a changed dataset")
    frozen.write_text(json.dumps(data, indent=2) + "\n")
    record = root / f"{model}.run.json"
    run = (
        json.loads(record.read_text())
        if record.exists()
        else {
            "model": model,
            "created_at": datetime.now(UTC).isoformat(),
            "clips": {},
            "failures": {},
            "timing_policy": "Loaded model, resampling + inference + native MIDI export; no network or model load.",
        }
    )
    worker = (
        YourMT3(architecture=model)
        if model in CHECKPOINTS
        else TransKun()
        if model == "transkun"
        else BasicPitch()
    )
    for e in examples:
        if e["id"] in run["clips"] and (
            model not in CHECKPOINTS or "export_policy" in run["clips"][e["id"]]
        ):
            continue
        audio = (manifest.parent / e["audio"]).read_bytes()
        if hashlib.sha256(audio).hexdigest() != e["audio_sha256"]:
            raise ValueError("Audio checksum mismatch")
        try:
            result = worker.transcribe.remote(audio)
            folder = root / e["id"]
            folder.mkdir(exist_ok=True)
            midi = result.pop("midi")
            (folder / f"{model}.mid").write_bytes(midi)
            if "demo_midi" in result:
                demo_midi = result.pop("demo_midi")
                (folder / f"{model}.demo.mid").write_bytes(demo_midi)
                result["demo_midi_sha256"] = hashlib.sha256(demo_midi).hexdigest()
            result["midi_sha256"] = hashlib.sha256(midi).hexdigest()
            run["clips"][e["id"]] = result
            run["failures"].pop(e["id"], None)
            print(
                f"{model} {len(run['clips'])}/{len(examples)} {e['id']}: {result['seconds']:.2f}s",
                flush=True,
            )
        except Exception as err:
            run["failures"][e["id"]] = f"{type(err).__name__}: {err}"
            raise
        finally:
            record.write_text(json.dumps(run, indent=2) + "\n")
    run["completed"] = all(e["id"] in run["clips"] for e in examples)
    run["inference_seconds"] = sum(c["seconds"] for c in run["clips"].values())
    record.write_text(json.dumps(run, indent=2) + "\n")

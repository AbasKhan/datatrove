#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

EXAMPLES_DIR = str(Path(__file__).resolve().parent)


from loguru import logger

from datatrove.data import Document, DocumentsPipeline
from datatrove.executor import SlurmPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter


CONTEXT_WINDOWS = (4096, 8192, 16384, 32768, 65536, 131072, 524288, 1048576)
BUCKET_LABELS = ("4k", "8k", "16k", "32k", "64k", "128k", "512k", "1m")
OVERFLOW_BUCKET_LABEL = "over_1m"


@dataclass(frozen=True)
class BucketSpec:
    name: str
    context_window: int | None
    min_tokens: int
    max_tokens: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bucket JSONL documents by token_count into 4k/8k/16k/32k/64k/128k/512k/1m/over_1m context buckets "
            "and launch the datatrove pipeline on Slurm."
        )
    )
    parser.add_argument("input_dir", help="Input directory containing JSONL files")
    parser.add_argument("output_dir", help="Output directory where bucketed JSONL files will be written")
    parser.add_argument("--partition", default="hopper-prod", help="Slurm partition to submit to")
    parser.add_argument("--tasks", type=int, default=10, help="Total number of datatrove tasks to launch")
    parser.add_argument("--time", default="30-00:00:00", help="Slurm time limit")
    parser.add_argument("--workers", type=int, default=10, help="Max concurrent Slurm array tasks (-1 means no limit)")
    parser.add_argument("--name", default="token_count_buckets", help="Base name used for Slurm jobs")
    parser.add_argument("--job-name", default=None, help="Slurm job name")
    parser.add_argument("--qos", default="low", help="Slurm QOS")
    parser.add_argument("--cpus-per-task", type=int, default=11, help="Slurm CPUs per task")
    parser.add_argument("--mem-per-cpu-gb", type=int, default=22, help="Slurm memory per CPU in GB")
    parser.add_argument(
        "--mem",
        default=None,
        help="Total Slurm memory like 128G; if set, mem_per_cpu_gb is derived from it like generate_data.py",
    )
    parser.add_argument("--gpus-per-task", type=int, default=0, help="Slurm GPUs per task")
    parser.add_argument("--nodes-per-task", type=int, default=1, help="Slurm nodes per task")
    parser.add_argument("--max-array-size", type=int, default=1001, help="Max Slurm array size for this cluster")
    parser.add_argument("--reservation", default=None, help="Optional Slurm reservation")
    parser.add_argument("--nodelist", default=None, help="Optional Slurm nodelist")
    parser.add_argument(
        "--logging-dir",
        default=None,
        help="Datatrove logging directory. Defaults to <output_dir>/_datatrove_logs",
    )
    parser.add_argument(
        "--slurm-logs-folder",
        default=None,
        help="Optional local directory for raw Slurm stdout/stderr logs",
    )
    parser.add_argument(
        "--glob-pattern",
        default="**/*.jsonl*",
        help="Glob pattern relative to input_dir used to select files",
    )
    parser.add_argument("--text-key", default="text", help="JSON key containing the document text")
    parser.add_argument("--id-key", default="id", help="JSON key containing the document id")
    parser.add_argument(
        "--token-count-key",
        default="token_count",
        help="Key used to read the token count from document metadata",
    )
    parser.add_argument(
        "--overhead-fraction",
        type=float,
        default=0.15,
        help="Safety margin subtracted from each context window before bucketing",
    )
    parser.add_argument(
        "--input-compression",
        choices=("infer", "gzip", "zstd", "none"),
        default="infer",
        help="Compression mode for reading input JSONL files",
    )
    parser.add_argument(
        "--output-compression",
        choices=("gzip", "zstd", "none"),
        default="none",
        help="Compression mode for output JSONL files",
    )
    parser.add_argument(
        "--output-filename-template",
        default="${context_bucket}_bucket/${rank}.jsonl",
        help=(
            "Writer template under output_dir. Must include ${rank}; ${context_bucket} is filled by the bucketizer "
            "so the default creates folders like 4k_bucket/ and 8k_bucket/"
        ),
    )
    parser.add_argument(
        "--env-command",
        default=None,
        help="Shell snippet to activate the runtime environment before starting each Slurm task",
    )
    parser.add_argument("--condaenv", default=None, help="Conda env name to activate inside Slurm jobs")
    parser.add_argument("--venv-path", default=None, help="Virtualenv path to activate inside Slurm jobs")
    parser.add_argument(
        "--skip-completed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip already completed ranks when relaunching",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recursively search input_dir for matching files",
    )
    return parser.parse_args()


def normalize_compression(compression: str) -> str | None:
    return None if compression == "none" else compression


def sanitize_job_name(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-")
    return sanitized or "token-count-buckets"


def build_default_slurm_env_command() -> str:
    hf_home = os.environ.get("HF_HOME") or str(REPO_ROOT / "hf_cache")
    site_packages_python = '$(python -c "import site; print(site.getsitepackages()[0])")'
    return (
        f"source {REPO_ROOT}/.venv/bin/activate"
        f" && export PYTHONPATH={SRC_DIR}:{EXAMPLES_DIR}:$PYTHONPATH"
        f' && export HF_HOME="{hf_home}"'
        ' && export HF_XET_CACHE="/tmp/hf_xet/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${SLURM_PROCID}"'
        ' && mkdir -p "$HF_XET_CACHE"'
        f' && export CUDA_HOME="{site_packages_python}/nvidia/cu13"'
        ' && export PATH="$CUDA_HOME/bin:$PATH"'
        f' && export LD_LIBRARY_PATH="$CUDA_HOME/lib:{site_packages_python}/nvidia/cuda_runtime/lib:${{LD_LIBRARY_PATH:-}}"'
        ' && export LIBRARY_PATH="$CUDA_HOME/lib:${LIBRARY_PATH:-}"'
        f' && export FLASHINFER_CACHE_DIR="{REPO_ROOT}/.venv/flashinfer_cache"'
    )


def resolve_mem_per_cpu_gb(mem: str | None, cpus_per_task: int, default_mem_per_cpu_gb: int) -> int:
    if not mem:
        return default_mem_per_cpu_gb

    digits = "".join(filter(str.isdigit, mem))
    if not digits:
        raise ValueError(f"Could not parse --mem value '{mem}'. Expected values like 128G.")
    total_mem_gb = int(digits)
    return max(1, total_mem_gb // max(1, cpus_per_task))


def build_bucket_specs(overhead_fraction: float) -> list[BucketSpec]:
    if not 0 <= overhead_fraction < 1:
        raise ValueError("overhead_fraction must be in the range [0, 1).")

    usable_fraction = 1.0 - overhead_fraction
    bucket_specs: list[BucketSpec] = []
    previous_max = -1

    for bucket_name, context_window in zip(BUCKET_LABELS, CONTEXT_WINDOWS, strict=True):
        max_tokens = math.floor(context_window * usable_fraction)
        bucket_specs.append(
            BucketSpec(
                name=bucket_name,
                context_window=context_window,
                min_tokens=previous_max + 1,
                max_tokens=max_tokens,
            )
        )
        previous_max = max_tokens

    bucket_specs.append(
        BucketSpec(
            name=OVERFLOW_BUCKET_LABEL,
            context_window=None,
            min_tokens=previous_max + 1,
            max_tokens=None,
        )
    )
    return bucket_specs


class TokenCountBucketizer(PipelineStep):
    name = "Token Count Bucketizer"
    type = "🪣 - BUCKETIZER"

    def __init__(self, bucket_specs: list[BucketSpec], token_count_key: str = "token_count"):
        super().__init__()
        self.bucket_specs = bucket_specs
        self.token_count_key = token_count_key
        self._warning_emitted = False

    def _parse_token_count(self, document: Document) -> int | None:
        raw_value = document.metadata.get(self.token_count_key)
        if raw_value is None or isinstance(raw_value, bool):
            return None

        try:
            token_count = int(raw_value)
        except (TypeError, ValueError):
            return None

        if token_count < 0:
            return None
        return token_count

    def _select_bucket(self, token_count: int) -> BucketSpec:
        for bucket in self.bucket_specs[:-1]:
            if bucket.max_tokens is not None and token_count <= bucket.max_tokens:
                return bucket
        return self.bucket_specs[-1]

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        from loguru import logger

        for document in data:
            token_count = self._parse_token_count(document)
            if token_count is None:
                self.stat_update("dropped_missing_or_invalid_token_count")
                if not self._warning_emitted:
                    self._warning_emitted = True
                    logger.warning(
                        "Dropping documents with missing or invalid '{}'. First bad document id: {}",
                        self.token_count_key,
                        document.id,
                    )
                continue

            bucket = self._select_bucket(token_count)
            document.metadata["context_bucket"] = bucket.name
            document.metadata["context_bucket_min_tokens"] = bucket.min_tokens
            document.metadata["context_bucket_max_tokens"] = bucket.max_tokens
            document.metadata["context_window_tokens"] = bucket.context_window
            document.metadata["bucketed_by_token_count_key"] = self.token_count_key
            self.stat_update(f"bucket_{bucket.name}")
            self.update_doc_stats(document)
            yield document


def main() -> None:
    args = parse_args()
    bucket_specs = build_bucket_specs(args.overhead_fraction)
    mem_per_cpu_gb = resolve_mem_per_cpu_gb(args.mem, args.cpus_per_task, args.mem_per_cpu_gb)
    env_command = args.env_command or build_default_slurm_env_command()

    job_name = args.job_name or sanitize_job_name(
        f"{args.name}-{Path(args.input_dir.rstrip('/')).name or 'input'}"
    )
    logging_dir = args.logging_dir or f"{args.output_dir.rstrip('/')}/_datatrove_logs"

    logger.info("Using bucket thresholds:")
    for bucket in bucket_specs:
        if bucket.max_tokens is None:
            logger.info("  {}: {}+ tokens", bucket.name, bucket.min_tokens)
        else:
            logger.info(
                "  {}: {}-{} tokens (context window {}, overhead {:.1%})",
                bucket.name,
                bucket.min_tokens,
                bucket.max_tokens,
                bucket.context_window,
                args.overhead_fraction,
            )

    pipeline = [
        JsonlReader(
            data_folder=args.input_dir,
            compression=normalize_compression(args.input_compression),
            glob_pattern=args.glob_pattern,
            text_key=args.text_key,
            id_key=args.id_key,
            recursive=args.recursive,
        ),
        TokenCountBucketizer(
            bucket_specs=bucket_specs,
            token_count_key=args.token_count_key,
        ),
        JsonlWriter(
            output_folder=args.output_dir,
            output_filename=args.output_filename_template,
            compression=normalize_compression(args.output_compression),
        ),
    ]

    executor = SlurmPipelineExecutor(
        pipeline=pipeline,
        tasks=args.tasks,
        workers=args.workers,
        job_name=job_name,
        time=args.time,
        partition=args.partition,
        max_array_launch_parallel=True,
        qos=args.qos,
        cpus_per_task=args.cpus_per_task,
        mem_per_cpu_gb=mem_per_cpu_gb,
        gpus_per_task=args.gpus_per_task,
        nodes_per_task=args.nodes_per_task,
        max_array_size=args.max_array_size,
        env_command=env_command,
        condaenv=args.condaenv,
        venv_path=args.venv_path,
        logging_dir=logging_dir,
        slurm_logs_folder=args.slurm_logs_folder,
        skip_completed=args.skip_completed,
        srun_args={"cpu-bind": "none"},
        sbatch_args={
            **({"reservation": args.reservation} if args.reservation else {}),
            **({"nodelist": args.nodelist} if args.nodelist else {}),
        },
    )
    executor.run()


if __name__ == "__main__":
    main()
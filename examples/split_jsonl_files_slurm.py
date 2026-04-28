#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import os
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

EXAMPLES_DIR = str(Path(__file__).resolve().parent)


from loguru import logger

from datatrove.executor import SlurmPipelineExecutor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split each input JSONL file into equally sized line-based parts and launch the datatrove pipeline on Slurm."
        )
    )
    parser.add_argument("input_dir", help="Input directory containing JSONL files")
    parser.add_argument("output_dir", help="Output directory where split JSONL files will be written")
    parser.add_argument("--partition", default="hopper-prod", help="Slurm partition to submit to")
    parser.add_argument("--tasks", type=int, default=10, help="Total number of datatrove tasks to launch")
    parser.add_argument("--time", default="30-00:00:00", help="Slurm time limit")
    parser.add_argument("--workers", type=int, default=10, help="Max concurrent Slurm array tasks (-1 means no limit)")
    parser.add_argument("--name", default="split_jsonl_files", help="Base name used for Slurm jobs")
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
    parser.add_argument("--n-splits", type=int, default=10, help="Number of equal line-based splits to create per input file")
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
        default="${source_rel_stem}_part${split_index_padded}.jsonl",
        help=(
            "Writer template under output_dir. Available metadata placeholders include ${source_rel_stem}, "
            "${source_filename}, ${split_index}, ${split_index_padded}, ${split_count}, and ${rank}."
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
    return sanitized or "split-jsonl-files"


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


def strip_jsonl_suffixes(path: Path) -> Path:
    stripped = path
    while stripped.suffix in {".gz", ".zst", ".jsonl"}:
        stripped = stripped.with_suffix("")
    return stripped


class EqualSplitJsonlReader(JsonlReader):
    name = "🐿 Jsonl Equal Split"

    def __init__(self, *args, n_splits: int, **kwargs):
        super().__init__(*args, **kwargs)
        if n_splits <= 0:
            raise ValueError("n_splits must be a positive integer.")
        self.n_splits = n_splits

    def _count_lines(self, filepath: str) -> int:
        with self.data_folder.open(filepath, "r", compression=self.compression) as file_handle:
            return sum(1 for _ in file_handle)

    def _build_split_metadata(self, filepath: str, line_index: int, chunk_size: int, total_lines: int) -> dict:
        relative_path = Path(filepath)
        if relative_path.is_absolute():
            try:
                relative_path = relative_path.relative_to(Path(self.data_folder.path))
            except ValueError:
                relative_path = Path(relative_path.name)

        relative_stem = strip_jsonl_suffixes(relative_path)
        split_index = (line_index // chunk_size) + 1
        emitted_splits = min(total_lines, self.n_splits)
        return {
            "source_filename": relative_path.name,
            "source_rel_path": relative_path.as_posix(),
            "source_rel_stem": relative_stem.as_posix(),
            "split_index": split_index,
            "split_index_padded": f"{split_index:02d}",
            "split_count": emitted_splits,
            "split_chunk_size": chunk_size,
            "source_total_lines": total_lines,
        }

    def read_file(self, filepath: str):
        import orjson
        from orjson import JSONDecodeError

        total_lines = self._count_lines(filepath)
        self.stat_update("input_lines", value=total_lines, unit="input_file")
        if total_lines == 0:
            self.stat_update("empty_input_files")
            logger.warning("Skipping empty file {}", filepath)
            return

        chunk_size = max(1, -(-total_lines // self.n_splits))
        emitted_splits = min(total_lines, self.n_splits)
        self.stat_update("splits_emitted", value=emitted_splits, unit="input_file")

        with self.data_folder.open(filepath, "r", compression=self.compression) as file_handle:
            try:
                for line_index, line in enumerate(file_handle):
                    with self.track_time():
                        try:
                            payload = orjson.loads(line)
                            for media in payload.get("media", []):
                                if media["media_bytes"] is not None:
                                    media["media_bytes"] = base64.decodebytes(media["media_bytes"].encode("ascii"))
                            document = self.get_document_from_dict(payload, filepath, line_index)
                            if not document:
                                continue
                        except (EOFError, JSONDecodeError) as error:
                            logger.warning(f"Error when reading `{filepath}`: {error}")
                            continue

                    document.metadata.update(
                        self._build_split_metadata(filepath, line_index, chunk_size, total_lines)
                    )
                    yield document
            except UnicodeDecodeError as error:
                logger.warning(f"File `{filepath}` may be corrupted: raised UnicodeDecodeError ({error})")


def main() -> None:
    args = parse_args()
    mem_per_cpu_gb = resolve_mem_per_cpu_gb(args.mem, args.cpus_per_task, args.mem_per_cpu_gb)
    env_command = args.env_command or build_default_slurm_env_command()

    job_name = args.job_name or sanitize_job_name(
        f"{args.name}-{Path(args.input_dir.rstrip('/')).name or 'input'}"
    )
    logging_dir = args.logging_dir or f"{args.output_dir.rstrip('/')}/_datatrove_logs"

    logger.info("Using equal file splits: {} parts per input file", args.n_splits)

    pipeline = [
        EqualSplitJsonlReader(
            data_folder=args.input_dir,
            compression=normalize_compression(args.input_compression),
            glob_pattern=args.glob_pattern,
            text_key=args.text_key,
            id_key=args.id_key,
            recursive=args.recursive,
            n_splits=args.n_splits,
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
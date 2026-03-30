"""
Script for generating synthetic data using vLLM inference. Uses the InferenceRunner
with chunking enabled. Documents are processed in chunks with checkpoint support
for resuming from failures. Each chunk is saved to a separate output file.

Supports local execution, SLURM clusters, and multi-node setups.

Usage:

# View all options
python examples/inference/generate_data.py --help

# Generate synthetic data locally using a prompt column
python examples/inference/generate_data.py \
    --input-dataset-name simplescaling/s1K-1.1 \
    --prompt-column question \
    --model-name-or-path Qwen/Qwen3-0.6B \
    --output-dataset-name s1K-1.1-datatrove \
    --tasks 1 \
    --workers 1 \
    --local-execution

# Generate synthetic data on a Slurm cluster
python examples/inference/generate_data.py \
    --input-dataset-name simplescaling/s1K-1.1 \
    --prompt-column question \
    --model-name-or-path Qwen/Qwen3-0.6B \
    --output-dataset-name s1K-1.1-benchmark

# Generate synthetic data using a prompt template with [[DOCUMENT]] variable
python examples/inference/generate_data.py \
    --input-dataset-name Salesforce/wikitext \
    --input-dataset-config wikitext-2-v1 \
    --prompt-column text \
    --prompt-template "Summarize the following document: [[DOCUMENT]]" \
    --model-name-or-path Qwen/Qwen3-0.6B \
    --output-dataset-name wikitext-summaries \
    --tasks 1 \
    --workers 1

# Generate synthetic data on multiple nodes
python examples/inference/generate_data.py \
    --input-dataset-name simplescaling/s1K-1.1 \
    --prompt-column question \
    --model-name-or-path moonshotai/Kimi-K2-Instruct \
    --model-max-context 1024 \
    --max-tokens 8 \
    --trust-remote-code \
    --output-dataset-name s1K-1.1-benchmark-Kimi-K2-Instruct \
    --tasks 1 \
    --workers 1 \
    --max-examples 100 \
    --nodes-per-task 2 \
    --tp 8 \
    --pp 2 \
    --optimization-level 0 \
    --max-num-seqs=16
"""

import os
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable

import typer

from datatrove.data import Document
from datatrove.pipeline.inference.dataset_card_generator import (
    InferenceDatasetCardGenerator,
    InferenceDatasetCardParams,
)
from datatrove.pipeline.inference.run_inference import InferenceConfig, InferenceResult, InferenceRunner
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.utils.logging import logger


# Add parent directory to path so utils can be imported
# This path is also exported in SLURM jobs for unpickling
EXAMPLES_INFERENCE_DIR = str(Path(__file__).parent)
sys.path.insert(0, EXAMPLES_INFERENCE_DIR)
from utils import (  # noqa: E402
    build_run_path,
    check_hf_auth,
    ensure_repo_exists,
    normalize_kvc_dtype,
    normalize_quantization,
    normalize_speculative,
    resolve_repo_id,
    validate_config,
)



def _compute_reader_limit(max_examples: int, tasks: int) -> int:
    """Compute per-rank reader limit so max_examples stays global.

    The HuggingFace reader applies `limit` on each rank independently. For
    multi-task runs this means `limit=max_examples` would multiply total output
    by the number of tasks. We split the global budget across tasks instead.
    """
    if max_examples <= 0:
        return max_examples
    if tasks < 1:
        raise ValueError("tasks must be >= 1 when max_examples is set.")
    reader_limit = (max_examples + tasks - 1) // tasks
    if tasks > 1:
        logger.info(f"Applying global max_examples={max_examples} across {tasks} tasks ({reader_limit} docs per task)")
    return reader_limit


def main(
    # Input data details
    input_dataset_name: str = "simplescaling/s1K-1.1",
    input_dataset_config: str | None = None,
    input_dataset_split: str = "train",
    prompt_column: str = "question",
    prompt_template: str | None = None,  # Can be "template", '["name", "template"]', or a named template
    max_examples: int = -1,
    # Output dataset details
    output_dataset_name: str = "s1K-1.1-datatrove",
    output_private: bool = True,
    # Output logs and tmp files
    output_dir: str = "data",
    # Inference settings
    server_type: str = "vllm",
    model_name_or_path: str = "Qwen/Qwen3-0.6B",
    model_revision: str = "main",
    model_max_context: int = 32768,
    system_prompt: str | None = None,
    # WARNING: Set to True only if you trust the model repository.
    # Enabling this allows execution of arbitrary code from the remote repository,
    # which can be a security risk. Use True only for trusted sources.
    trust_remote_code: bool = False,
    # vLLM distribution settings
    tp: int = 1,
    pp: int = 1,
    dp: int = 1,
    nodes_per_task: int = 1,
    # vLLM server settings (there should be no need to change the defaults)
    max_concurrent_generations: int = 500,
    max_concurrent_documents: int = 500,
    max_num_seqs: int = 256,  # reduce this if you run out of memory
    max_num_batched_tokens: int = 8192,  # controls chunked prefill batch size
    gpu_memory_utilization: float = 0.9,  # Fraction of GPU memory for KV cache
    block_size: int = 16,  # KV cache block size (16 or 32)
    speculative_config: str | None = None,
    quantization: str | None = None,  # "bitsandbytes" for 4-bit quantization
    kv_cache_dtype: str = "auto",  # "auto", "fp8_e4m3", or "fp8_e5m2"
    optimization_level: int = 3,  # Set to 0 for fastest startup, 3 for best throughput
    metric_interval: int = 120,
    # Generation parameters
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    min_p: float | None = None,
    presence_penalty: float | None = None,
    repetition_penalty: float | None = None,
    max_tokens: int = 16384,
    enable_thinking: bool = True,  # Set to False to disable thinking for models like Qwen3
    rollouts_per_document: int = 1,
    seed: int | None = None,  # Random seed for reproducible generation
    # Processing settings
    examples_per_chunk: int = 500,
    tasks: int = 10,
    workers: int = 10,
    local_execution: bool = False,
    enable_monitoring: bool = False,
    benchmark_mode: bool = False,  # Skip output writing for benchmarking
    # slurm settings
    name: str = "synth",
    time: str = "12:00:00",
    qos: str = "low",
    partition: str = "hopper-prod",
    account: str | None = None,
    reservation: str | None = None,
    mem: str | None = None,  # Total memory e.g. "128G"; overrides mem-per-cpu default
) -> None:
    """Typer CLI entrypoint that runs the pipeline with provided options."""
    output_dataset_path = Path(output_dataset_name).expanduser()
    save_output_locally = output_dataset_path.is_absolute() or output_dataset_name.startswith(".")

    # Skip HuggingFace setup in benchmark mode or when writing output locally
    full_repo_id = None
    if benchmark_mode:
        enable_monitoring = False
    elif not save_output_locally:
        check_hf_auth()  # Check authentication early to avoid errors later
        full_repo_id = resolve_repo_id(output_dataset_name)  # Resolve full repo name for the output dataset
        ensure_repo_exists(full_repo_id, private=output_private)  # Create the repository if it doesn't exist

    if local_execution:
        import torch

        available_gpus = torch.cuda.device_count()
        if available_gpus == 0:
            raise ValueError("Local execution requires at least one CUDA GPU.")
        tp = min(tp, available_gpus)
        pp = 1
        nodes_per_task = 1
        logger.info(f"Local execution on {available_gpus} GPUs on one node")


    # Parse prompt_template: can be a named template (e.g. "faq"), a file path, a JSON ["name", "template"] list,
    # or a raw template string containing [[DOCUMENT]]
    _parsed_template: str | list | None = prompt_template
    if isinstance(prompt_template, str):
        import json

        template_path = Path(prompt_template).expanduser()
        if template_path.is_file():
            _parsed_template = [template_path.stem, template_path.read_text()]
        elif prompt_template.startswith("["):
            _parsed_template = json.loads(prompt_template)
        else:
            try:
                from finephrase import PROMPT_TEMPLATES  # noqa: PLC0415

                if prompt_template in PROMPT_TEMPLATES:
                    _parsed_template = [prompt_template, PROMPT_TEMPLATES[prompt_template]]
            except ImportError:
                pass
    prompt_template_name, prompt_template = (
        _parsed_template if isinstance(_parsed_template, list) else ("default", _parsed_template)
    )

    gpus_per_node = validate_config(
        tp=tp,
        pp=pp,
        dp=dp,
        nodes_per_task=nodes_per_task,
        optimization_level=optimization_level,
        config=None,
        prompt_template=prompt_template,
    )

    async def simple_rollout(
        document: Document,
        generate: Callable[[dict[str, Any]], Awaitable[InferenceResult]],
    ) -> InferenceResult:
        """
        Basic rollout that sends a single request per document.

        Returns the InferenceResult directly, which will be stored under document.metadata["rollout_results"].
        """
        messages = [] if system_prompt is None else [{"role": "system", "content": system_prompt}]

        if isinstance(document.text, list) and all(isinstance(msg, dict) for msg in document.text):
            if prompt_template:
                raise ValueError("Prompt template is not supported for message lists")
            messages.extend(document.text)
        else:
            content = prompt_template.replace("[[DOCUMENT]]", document.text) if prompt_template else document.text

            # Truncate content if too long to avoid server errors
            CHAR_PER_TOKEN = 3  # Uses ~3 chars per token as a conservative approximation
            char_budget = int((model_max_context - max_tokens) * CHAR_PER_TOKEN)
            if len(content) > char_budget:
                original_len = len(content)
                # Try to truncate at a newline boundary for cleaner cuts
                last_newline = content.rfind("\n", 0, char_budget)
                content = content[:last_newline] if last_newline != -1 else content[:char_budget]
                # Import logger inside the function to ensure it's available in pickled closures
                from datatrove.utils.logging import logger as _logger

                _logger.warning(
                    f"Truncated content from {original_len} to {len(content)} chars (budget: {char_budget} chars)"
                )

            messages.append({"role": "user", "content": content})

        return await generate(
            {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                **({"min_p": min_p} if min_p is not None else {}),
                **({"presence_penalty": presence_penalty} if presence_penalty is not None else {}),
                **({"repetition_penalty": repetition_penalty} if repetition_penalty is not None else {}),
                **({"seed": seed} if seed is not None else {}),
                "chat_template_kwargs": {"enable_thinking": enable_thinking},
            }
        )

    async def chunked_rollout(
        document: Document,
        generate: Callable[[dict[str, Any]], Awaitable[InferenceResult]],
    ) -> InferenceResult:
        """
        Rollout that splits long documents into chunks and generates sequentially,
        using the previous generation as an assistant prefill for each subsequent chunk.
        """
        CHAR_PER_TOKEN = 3
        max_chars_per_part = int((model_max_context - max_tokens) * CHAR_PER_TOKEN)

        if isinstance(document.text, list) and all(isinstance(msg, dict) for msg in document.text):
            raise ValueError("Chunked rollout does not support message list inputs")

        content = prompt_template.replace("[[DOCUMENT]]", document.text) if prompt_template else document.text
        chunks = [content[i : i + max_chars_per_part] for i in range(0, len(content), max_chars_per_part)] or [content]

        generations: list[str] = []
        last_result: InferenceResult | None = None
        combined_usage: dict = {}

        for chunk in chunks:
            messages = [] if system_prompt is None else [{"role": "system", "content": system_prompt}]
            prev = generations[-1] if generations else ""
            messages.append({"role": "user", "content": chunk})
            messages.append({"role": "assistant", "content": prev})

            last_result = await generate(
                {
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": top_p,
                    **({"min_p": min_p} if min_p is not None else {}),
                    **({"presence_penalty": presence_penalty} if presence_penalty is not None else {}),
                    **({"repetition_penalty": repetition_penalty} if repetition_penalty is not None else {}),
                    **({"seed": seed} if seed is not None else {}),
                    "chat_template_kwargs": {"enable_thinking": enable_thinking},
                    "continue_final_message": True,
                }
            )
            generations.append(last_result.text)
            for key in ("completion_tokens", "prompt_tokens", "total_tokens"):
                combined_usage[key] = combined_usage.get(key, 0) + last_result.usage.get(key, 0)

        return InferenceResult(
            text="".join(generations),
            finish_reason=last_result.finish_reason,
            usage=combined_usage,
        )

    temperature = temperature if temperature is not None else 1.0
    top_p = top_p if top_p is not None else 1.0
    top_k = top_k if top_k is not None else -1

    # Normalize speculative config; treat common "none" strings as disabled
    spec_raw = speculative_config
    if isinstance(spec_raw, str) and spec_raw.strip().lower() in ("none", "null", ""):
        spec_raw = None
    normalized_spec = normalize_speculative(spec_raw)

    # Normalize quantization and KV cache dtype configs
    normalized_quant = normalize_quantization(quantization)
    normalized_kv_dtype = normalize_kvc_dtype(kv_cache_dtype)

    # Build dynamic output directory: {output_dir}/{prompt}/{model}/{tp-pp-dp}/{mns}/{mnbt}/{gmu}/{bs}/{kvc}/{spec}/{quant}
    run_path = build_run_path(
        output_dir=output_dir,
        prompt_template_name=prompt_template_name,
        model_name_or_path=model_name_or_path,
        tp=tp,
        pp=pp,
        dp=dp,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        block_size=block_size,
        kv_cache_dtype=kv_cache_dtype,
        speculative_config=spec_raw,
        quantization=quantization,
    )
    if benchmark_mode:
        output_path = str(run_path / "output" / "data")
    elif save_output_locally:
        output_path = str(output_dataset_path / prompt_template_name)
    else:
        output_path = f"hf://datasets/{full_repo_id}/{prompt_template_name}"
    checkpoints_path = str(run_path / "checkpoints")
    inference_logs_path = run_path / "inference_logs"
    monitor_logs_path = run_path / "monitor_logs"
    datacard_logs_path = run_path / "datacard_logs"

    # Build quantization-specific kwargs for vLLM
    quant_kwargs: dict[str, Any] = {}
    if normalized_quant == "bitsandbytes":
        # BitsAndBytes 4-bit quantization
        quant_kwargs["quantization"] = "bitsandbytes"

    # Build KV cache dtype kwargs for vLLM
    kv_cache_kwargs: dict[str, Any] = {}
    if normalized_kv_dtype != "auto":
        # FP8 KV cache (reduces memory while maintaining quality)
        kv_cache_kwargs["kv_cache_dtype"] = normalized_kv_dtype
        kv_cache_kwargs["calculate_kv_scales"] = True

    _model_kwargs = {
        "revision": model_revision,
        "dtype": "bfloat16",
        "max_num_seqs": max_num_seqs,
        "max_num_batched_tokens": max_num_batched_tokens,
        "block-size": block_size,
        "gpu-memory-utilization": gpu_memory_utilization,
        **({"speculative_config": normalized_spec} if normalized_spec else {}),
        **quant_kwargs,
        **kv_cache_kwargs,
        "optimization-level": optimization_level,
    }
    # Memory per CPU for slurm allocation (in GB)
    # If --mem total memory is given, derive mem_per_cpu_gb from it; otherwise use default
    if mem:
        mem_gb = int("".join(filter(str.isdigit, mem)))
        cpus_per_task = gpus_per_node * 11
        mem_per_cpu_gb = max(1, mem_gb // cpus_per_task)
    else:
        mem_per_cpu_gb = 22
    if not local_execution and nodes_per_task > 1:
        # vLLM defaults to the mp backend when TP fits on a single host; but when TP spans
        # multiple nodes we must force the Ray backend so TP can exceed local GPU count.
        _model_kwargs["distributed-executor-backend"] = "ray"
        # Help any Ray client in subprocesses (like `vllm serve`) attach to the running cluster.
        os.environ["RAY_ADDRESS"] = "auto"

    inference_config = InferenceConfig(
        server_type=server_type,
        model_name_or_path=model_name_or_path,
        model_kwargs=_model_kwargs,
        model_max_context=model_max_context,
        rollouts_per_document=rollouts_per_document,
        max_concurrent_generations=max_concurrent_generations,
        max_concurrent_documents=max_concurrent_documents,
        metric_interval=metric_interval,
        tp=tp,
        dp=dp,
        pp=pp,
        server_log_folder=str(inference_logs_path / "server_logs"),
    )

    input_dataset_path = Path(input_dataset_name).expanduser()
    load_input_from_disk = input_dataset_path.exists() and input_dataset_path.is_dir()

    if load_input_from_disk and input_dataset_config is not None:
        logger.warning("Ignoring --input-dataset-config because local datasets loaded from disk do not use configs.")

    if load_input_from_disk:
        logger.info(f"Loading input dataset from local disk: {input_dataset_path}")

    inference_pipeline = [
        HuggingFaceDatasetReader(
            dataset="parquet" if load_input_from_disk else input_dataset_name,
            dataset_options={"data_dir": str(input_dataset_path), "split": "train"} if load_input_from_disk else {"name": input_dataset_config, "split": input_dataset_split},
            text_key=prompt_column,
            limit=_compute_reader_limit(max_examples=max_examples, tasks=tasks),
            load_from_disk=False,
        ),
        InferenceRunner(
            rollout_fn=simple_rollout,
            config=inference_config,
            records_per_chunk=examples_per_chunk,
            checkpoints_local_dir=checkpoints_path if not benchmark_mode else None,
            skip_bad_requests=True,
            # The HuggingFaceDatasetWriter only uploads at the end, the ParquetWriter uploads incrementally
            output_writer=JsonlWriter(
                output_folder=output_path,
                output_filename="${rank}_${chunk_index}.jsonl",
                compression=None,
                expand_metadata=True,
            ),
        ),
    ]

    dataset_card_params = None
    datacard_pipeline = None
    if not save_output_locally and not benchmark_mode:
        dataset_card_params = InferenceDatasetCardParams(
            output_repo_id=full_repo_id,
            input_dataset_name=input_dataset_name,
            input_dataset_split=input_dataset_split,
            input_dataset_config=input_dataset_config,
            prompt_column=prompt_column,
            prompt_template=prompt_template,
            prompt_template_name=prompt_template_name,
            system_prompt=system_prompt,
            model_name=model_name_or_path,
            model_revision=model_revision,
            generation_kwargs={
                "max_tokens": max_tokens,
                "model_max_context": model_max_context,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "seed": seed,
            },
            spec_config=normalized_spec,
            stats_path=str(inference_logs_path / "stats.json"),
        )
        datacard_pipeline = [InferenceDatasetCardGenerator(params=dataset_card_params)]

    if local_execution:
        from datatrove.executor import LocalPipelineExecutor  # Lazy import to speed up startup time

        inference_executor = LocalPipelineExecutor(
            pipeline=inference_pipeline,
            logging_dir=str(inference_logs_path),
            tasks=tasks,
            workers=workers,
        )
        inference_executor.run()

        if datacard_pipeline is not None:
            datacard_executor = LocalPipelineExecutor(
                pipeline=datacard_pipeline,
                logging_dir=str(datacard_logs_path),
                tasks=1,
                workers=1,
            )
            # Monitor not supported in local execution as it would block
            datacard_executor.run()
    else:
        from datatrove.executor import SlurmPipelineExecutor  # Lazy import to speed up startup time

        # Isolate Xet cache per Slurm process to avoid cache contention across parallel jobs.
        hf_home = os.environ.get("HF_HOME") or str(Path(__file__).parent.parent.parent / "hf_cache")
        slurm_env_command = (
            f"source .venv/bin/activate && export PYTHONPATH={EXAMPLES_INFERENCE_DIR}:$PYTHONPATH"
            f' && export HF_HOME="{hf_home}"'
            ' && export HF_XET_CACHE="/tmp/hf_xet/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${SLURM_PROCID}"'
            ' && mkdir -p "$HF_XET_CACHE"'
            " && export CUDA_HOME=/software/genoa/r25.06/CUDA/12.8.0"
            " && export PATH=$CUDA_HOME/bin:$PATH"
        )

        inference_executor = SlurmPipelineExecutor(
            pipeline=inference_pipeline,
            logging_dir=str(inference_logs_path),
            tasks=tasks,
            workers=workers,
            time=time,
            partition=partition,
            max_array_launch_parallel=True,
            qos=qos,
            job_name=f"{name}_inference",
            cpus_per_task=gpus_per_node * 11,
            mem_per_cpu_gb=mem_per_cpu_gb,
            # Required so Datatrove starts Ray with GPUs; otherwise it launches Ray with `--num-gpus 0`.
            gpus_per_task=gpus_per_node,
            nodes_per_task=nodes_per_task,
            srun_args={"cpu-bind": "none"},
            sbatch_args={
                **({"reservation": reservation} if reservation else {}),
                **({"account": account} if account else {}),
            },
            env_command=slurm_env_command,
        )
        inference_executor.run()

        if enable_monitoring:
            # Lazy import to speed up startup time
            from datatrove.pipeline.inference.progress_monitor import InferenceProgressMonitor

            monitor_pipeline = [
                InferenceProgressMonitor(
                    params=dataset_card_params,
                    max_examples=max_examples,
                    update_interval=60 if local_execution else 3600,  # 1 minute for debugging, 1 hour for slurm
                )
            ]
            # Update monitor with inference job id so it can stop if inference fails
            monitor_pipeline[0].inference_job_id = inference_executor.job_id

            monitor_executor = SlurmPipelineExecutor(
                pipeline=monitor_pipeline,
                logging_dir=str(monitor_logs_path),
                tasks=1,
                workers=1,
                time="7-00:00:00",  # Long enough to outlast inference
                partition=partition,
                qos=qos,
                job_name=f"{name}_monitor",
                cpus_per_task=1,
                sbatch_args={"mem-per-cpu": "4G", "requeue": "", **({"account": account} if account else {})},
                env_command=slurm_env_command,
            )

            monitor_executor.run()

        if datacard_pipeline is not None:
            datacard_executor = SlurmPipelineExecutor(
                pipeline=datacard_pipeline,
                logging_dir=str(datacard_logs_path),
                tasks=1,
                workers=1,
                time="0:10:00",
                partition=partition,
                qos=qos,
                job_name=f"{name}_datacard",
                cpus_per_task=1,
                depends=inference_executor,
                run_on_dependency_fail=False,  # use afterok
                sbatch_args={"mem-per-cpu": "4G", **({"account": account} if account else {})},
                env_command=slurm_env_command,
            )
            datacard_executor.run()

    return inference_executor.job_id


if __name__ == "__main__":
    typer.run(main)

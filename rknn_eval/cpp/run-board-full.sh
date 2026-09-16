#!/usr/bin/env sh
set -eu

RUNNER=/userdata/ptq-bench-rknn/rknn_llm_ppl_eval
DATA_ROOT=/userdata/ptq-bench-rknn/data
RESULT_ROOT=/userdata/ptq-bench-rknn/results
STATUS_FILE=${RESULT_ROOT}/full_eval.status
PID_FILE=/userdata/ptq-bench-rknn/full_eval.pid
PERF_PREFILL_LENGTHS=${PERF_PREFILL_LENGTHS:-"128 512 1024"}
RUN_WIKITEXT=${RUN_WIKITEXT:-1}
RUN_C4=${RUN_C4:-1}

OFFICIAL_ROOT=/userdata/llm_demo/rknn_Qwen3_5_demo/model_4b
Q2N_ROOT=/userdata/llm_demo/rknn_Qwen3_5_demo/model_4b_q2n_w4a16_g32

finish() {
  exit_code=$?
  if [ "${exit_code}" -eq 0 ]; then
    printf 'complete\n' > "${STATUS_FILE}"
  else
    printf 'failed:%s\n' "${exit_code}" > "${STATUS_FILE}"
  fi
  date '+finished_at=%Y-%m-%dT%H:%M:%S%z'
}
trap finish EXIT

mkdir -p "${RESULT_ROOT}"
printf '%s\n' "$$" > "${PID_FILE}"
printf 'running\n' > "${STATUS_FILE}"
date '+started_at=%Y-%m-%dT%H:%M:%S%z'

run_perf() {
  model_name=$1
  model_root=$2
  model_stem=$3
  prefill=$4
  output=${RESULT_ROOT}/${model_name}_perf_p${prefill}_d128.jsonl
  if [ -s "${output}.summary.json" ]; then
    echo "phase=performance_skip model=${model_name} prefill=${prefill} decode=128"
    return
  fi
  echo "phase=performance model=${model_name} prefill=${prefill} decode=128"
  "${RUNNER}" \
    --model "${model_root}/${model_stem}.rknn" \
    --weight "${model_root}/${model_stem}.weight" \
    --tokenizer "${model_root}/${model_stem}.tokenizer.gguf" \
    --embedding "${model_root}/${model_stem}.embed.bin" \
    --data "${DATA_ROOT}/wikitext2_ppl_seq1024.jsonl" \
    --output "${output}" \
    --model-name "${model_name}" \
    --logits-name logits --core-mask 0xff --max-context-len 4096 \
    --scoring-threads 4 --perf-prefill-tokens "${prefill}" \
    --perf-decode-tokens 128 --perf-warmup 3 --perf-repeat 10 \
    --limit 0 --no-resume
}

run_ppl() {
  model_name=$1
  model_root=$2
  model_stem=$3
  dataset=$4
  output=${RESULT_ROOT}/${model_name}_${dataset}.jsonl
  echo "phase=perplexity model=${model_name} dataset=${dataset}"
  "${RUNNER}" \
    --model "${model_root}/${model_stem}.rknn" \
    --weight "${model_root}/${model_stem}.weight" \
    --tokenizer "${model_root}/${model_stem}.tokenizer.gguf" \
    --embedding "${model_root}/${model_stem}.embed.bin" \
    --data "${DATA_ROOT}/${dataset}.jsonl" \
    --output "${output}" \
    --model-name "${model_name}" \
    --logits-name logits --core-mask 0xff --max-context-len 4096 \
    --scoring-threads 4 --perf-prefill-tokens 0 --perf-decode-tokens 0
}

for prefill in ${PERF_PREFILL_LENGTHS}; do
  run_perf Qwen3.5-4B-rknn-official "${OFFICIAL_ROOT}" Qwen3.5-4B "${prefill}"
done

for prefill in ${PERF_PREFILL_LENGTHS}; do
  run_perf Qwen3.5-4B-Q2N-W4A16-G32 "${Q2N_ROOT}" Qwen3.5-4B-Q2N-W4A16-G32 "${prefill}"
done

if [ "${RUN_WIKITEXT}" = 1 ]; then
  run_ppl Qwen3.5-4B-rknn-official "${OFFICIAL_ROOT}" Qwen3.5-4B wikitext2_ppl_seq1024
  run_ppl Qwen3.5-4B-Q2N-W4A16-G32 "${Q2N_ROOT}" Qwen3.5-4B-Q2N-W4A16-G32 wikitext2_ppl_seq1024
fi

if [ "${RUN_C4}" = 1 ]; then
  run_ppl Qwen3.5-4B-rknn-official "${OFFICIAL_ROOT}" Qwen3.5-4B c4_ppl_256x1024
  run_ppl Qwen3.5-4B-Q2N-W4A16-G32 "${Q2N_ROOT}" Qwen3.5-4B-Q2N-W4A16-G32 c4_ppl_256x1024
fi

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <omp.h>

#include "Tokenizer.h"
#include "nlohmann/json.hpp"
#include "rknn3_api.h"

using json = nlohmann::json;
using Clock = std::chrono::steady_clock;

namespace {

struct Options {
  std::string model;
  std::string weight;
  std::string tokenizer;
  std::string embedding;
  std::string data;
  std::string output;
  std::string model_name;
  std::string logits_name = "logits";
  std::string adapter = "generic_text";
  uint32_t core_mask = 0xff;
  int max_context_len = 1024;
  int limit = -1;
  int skip = 0;
  int scoring_threads = 4;
  int perf_prefill_tokens = 512;
  int perf_decode_tokens = 128;
  int perf_warmup = 0;
  int perf_repeat = 1;
  bool no_resume = false;
};

struct EmbeddingTable {
  int fd = -1;
  void* mapping = MAP_FAILED;
  size_t size = 0;
  const float16* data = nullptr;
  int vocab_size = 0;
  int embedding_dim = 0;

  ~EmbeddingTable() { close(); }

  void open(const std::string& path, int vocab, int dim) {
    if (path.empty() || path == "-") return;
    fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open embedding: " + path);
    struct stat status {};
    if (fstat(fd, &status) != 0) throw std::runtime_error("cannot stat embedding: " + path);
    const uint64_t expected = static_cast<uint64_t>(vocab) * dim * sizeof(float16);
    if (static_cast<uint64_t>(status.st_size) != expected) {
      std::ostringstream message;
      message << "embedding size mismatch: expected " << expected << ", got " << status.st_size;
      throw std::runtime_error(message.str());
    }
    size = static_cast<size_t>(status.st_size);
    mapping = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapping == MAP_FAILED) throw std::runtime_error("cannot mmap embedding: " + path);
    data = static_cast<const float16*>(mapping);
    vocab_size = vocab;
    embedding_dim = dim;
  }

  bool enabled() const { return data != nullptr; }

  void close() {
    if (mapping != MAP_FAILED) munmap(mapping, size);
    if (fd >= 0) ::close(fd);
    fd = -1;
    mapping = MAP_FAILED;
    data = nullptr;
    size = 0;
  }
};

struct HostMemory {
  uint64_t rss_bytes = 0;
  uint64_t peak_rss_bytes = 0;
};

struct Scorer {
  const std::vector<int32_t>* targets = nullptr;
  size_t index = 0;
  int vocab_size = 0;
  double nll_sum = 0.0;
  double nll_compute_seconds = 0.0;
  Clock::time_point run_started;
  Clock::time_point first_logits;
  bool saw_first_logits = false;
  std::string error;

  void reset(const std::vector<int32_t>* value, int vocab) {
    targets = value;
    index = 0;
    vocab_size = vocab;
    nll_sum = 0.0;
    nll_compute_seconds = 0.0;
    saw_first_logits = false;
    error.clear();
  }
};

struct Runtime {
  rknn3_context context = 0;
  rknn3_session* session = nullptr;
  Tokenizer* tokenizer = nullptr;
  EmbeddingTable embedding;
  Scorer scorer;
  int max_context_len = 0;

  ~Runtime() {
    if (session) rknn3_session_destroy(session);
    if (context) rknn3_destroy(context);
    delete tokenizer;
  }
};

double seconds_between(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration<double>(end - begin).count();
}

uint64_t parse_size_kb_line(const std::string& line) {
  std::istringstream stream(line);
  std::string key;
  uint64_t value = 0;
  std::string unit;
  stream >> key >> value >> unit;
  return value * 1024;
}

HostMemory read_host_memory() {
  HostMemory result;
  std::ifstream input("/proc/self/status");
  std::string line;
  while (std::getline(input, line)) {
    if (line.compare(0, 6, "VmRSS:") == 0) result.rss_bytes = parse_size_kb_line(line);
    if (line.compare(0, 6, "VmHWM:") == 0) result.peak_rss_bytes = parse_size_kb_line(line);
  }
  return result;
}

json host_memory_json(const HostMemory& memory) {
  return {{"rss_bytes", memory.rss_bytes}, {"peak_rss_bytes", memory.peak_rss_bytes}};
}

json query_device_memory(rknn3_context context) {
  rknn3_dev_mem_info info {};
  const int ret = rknn3_query(context, RKNN3_QUERY_DEVICE_MEM_INFO, &info, sizeof(info));
  if (ret < 0) return {{"available", false}, {"ret", ret}};
  json nodes = json::array();
  for (uint32_t i = 0; i < info.node_num; ++i) {
    nodes.push_back({{"index", i}, {"total_bytes", info.node_mem_info[i].total},
                     {"free_bytes", info.node_mem_info[i].free}});
  }
  return {{"available", true}, {"sys_total_bytes", info.sys_total},
          {"sys_free_bytes", info.sys_free}, {"nodes", nodes}};
}

uint32_t query_core_count(rknn3_context context) {
  uint32_t count = 0;
  const int ret = rknn3_query(context, RKNN3_QUERY_CORE_NUMBER, &count, sizeof(count));
  if (ret < 0) throw std::runtime_error("RKNN3_QUERY_CORE_NUMBER failed: " + std::to_string(ret));
  return count;
}

json query_allocations(rknn3_context context, uint32_t core_count) {
  std::vector<rknn3_allocation_info> info(core_count);
  for (uint32_t i = 0; i < core_count; ++i) info[i].core_id = static_cast<int32_t>(i);
  const int ret = rknn3_query(context, RKNN3_QUERY_ALLOCATION_INFO, info.data(),
                              info.size() * sizeof(info[0]));
  if (ret < 0) return {{"available", false}, {"ret", ret}};
  uint64_t command = 0, weight = 0, internal = 0, kvcache = 0;
  json cores = json::array();
  for (const auto& item : info) {
    command += item.command_mem.size;
    weight += item.weight_mem.size;
    internal += item.internal_mem.size;
    kvcache += item.kvcache_mem.size;
    cores.push_back({{"core_id", item.core_id},
                     {"command_bytes", item.command_mem.size},
                     {"weight_bytes", item.weight_mem.size},
                     {"internal_bytes", item.internal_mem.size},
                     {"kvcache_bytes", item.kvcache_mem.size}});
  }
  return {{"available", true}, {"command_bytes", command}, {"weight_bytes", weight},
          {"internal_bytes", internal}, {"kvcache_bytes", kvcache},
          {"total_bytes", command + weight + internal + kvcache}, {"cores", cores}};
}

json query_kvcache_groups(rknn3_context context, uint32_t core_count) {
  std::vector<rknn3_kvcache_len_group_info> info(core_count);
  for (uint32_t i = 0; i < core_count; ++i) info[i].core_id = static_cast<int32_t>(i);
  const int ret = rknn3_query(context, RKNN3_QUERY_KVCACHE_LEN_GROUP_INFO, info.data(),
                              info.size() * sizeof(info[0]));
  if (ret < 0) return {{"available", false}, {"ret", ret}};
  json cores = json::array();
  for (const auto& item : info) {
    json sizes = json::array();
    for (uint32_t i = 0; i < item.n_groups; ++i) sizes.push_back(item.kvcache_sizes[i]);
    cores.push_back({{"core_id", item.core_id}, {"group_count", item.n_groups},
                     {"active_group_id", item.active_group_id},
                     {"group_sizes_bytes", sizes}});
  }
  return {{"available", true}, {"cores", cores}};
}

int result_callback(void*, RKLLMResult*, LLMCallState state) {
  return state == RKLLM_RUN_ERROR ? -1 : 0;
}

int embed_callback(void* userdata, int32_t* tokens, uint64_t num_tokens,
                   void* output, uint64_t len) {
  EmbeddingTable* table = static_cast<EmbeddingTable*>(userdata);
  const uint64_t expected = num_tokens * table->embedding_dim * sizeof(float16);
  if (!table->enabled() || len != expected) return -1;
  for (uint64_t i = 0; i < num_tokens; ++i) {
    if (tokens[i] < 0 || tokens[i] >= table->vocab_size) return -1;
    std::memcpy(static_cast<uint8_t*>(output) + i * table->embedding_dim * sizeof(float16),
                table->data + static_cast<uint64_t>(tokens[i]) * table->embedding_dim,
                table->embedding_dim * sizeof(float16));
  }
  return 0;
}

int sampling_callback(void* userdata, float16* logits, char*) {
  Scorer* scorer = static_cast<Scorer*>(userdata);
  if (!scorer->targets || scorer->index >= scorer->targets->size()) {
    scorer->error = "sampling callback received more logits than targets";
    return -1;
  }
  const auto started = Clock::now();
  if (!scorer->saw_first_logits) {
    scorer->first_logits = started;
    scorer->saw_first_logits = true;
  }
  const int target = (*scorer->targets)[scorer->index];
  if (target < 0 || target >= scorer->vocab_size) {
    scorer->error = "target token is outside vocabulary";
    return -1;
  }
  float maximum = -std::numeric_limits<float>::infinity();
#pragma omp parallel for reduction(max : maximum)
  for (int i = 0; i < scorer->vocab_size; ++i) {
    maximum = std::max(maximum, fp16_to_fp32(logits[i]));
  }
  double sum = 0.0;
#pragma omp parallel for reduction(+ : sum)
  for (int i = 0; i < scorer->vocab_size; ++i) {
    sum += std::exp(static_cast<double>(fp16_to_fp32(logits[i]) - maximum));
  }
  scorer->nll_sum += static_cast<double>(maximum) + std::log(sum) -
                     static_cast<double>(fp16_to_fp32(logits[target]));
  ++scorer->index;
  scorer->nll_compute_seconds += seconds_between(started, Clock::now());
  return target;
}

uint32_t parse_u32(const std::string& value) {
  char* end = nullptr;
  errno = 0;
  const unsigned long parsed = std::strtoul(value.c_str(), &end, 0);
  if (errno || !end || *end != '\0' || parsed > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("invalid unsigned integer: " + value);
  }
  return static_cast<uint32_t>(parsed);
}

int parse_int(const std::string& value, const std::string& option) {
  char* end = nullptr;
  errno = 0;
  const long parsed = std::strtol(value.c_str(), &end, 10);
  if (errno || !end || *end != '\0' || parsed < std::numeric_limits<int>::min() ||
      parsed > std::numeric_limits<int>::max()) {
    throw std::invalid_argument("invalid value for " + option + ": " + value);
  }
  return static_cast<int>(parsed);
}

void usage(const char* program) {
  std::cerr << "Usage: " << program
            << " --model M --weight W --tokenizer T --embedding E --data D --output O"
               " [--model-name N] [--logits-name logits] [--adapter generic_text]"
               " [--core-mask 0xff] [--max-context-len 1024] [--limit N] [--skip N]"
               " [--scoring-threads 4] [--perf-prefill-tokens 512]"
               " [--perf-decode-tokens 128] [--perf-warmup 0] [--perf-repeat 1]"
               " [--no-resume]\n"
               "Use --embedding - for models with tied token embeddings.\n";
}

Options parse_options(int argc, char** argv) {
  Options options;
  std::map<std::string, std::string*> strings = {
      {"--model", &options.model}, {"--weight", &options.weight},
      {"--tokenizer", &options.tokenizer}, {"--embedding", &options.embedding},
      {"--data", &options.data}, {"--output", &options.output},
      {"--model-name", &options.model_name}, {"--logits-name", &options.logits_name},
      {"--adapter", &options.adapter}};
  for (int i = 1; i < argc; ++i) {
    const std::string key(argv[i]);
    if (key == "--no-resume") {
      options.no_resume = true;
      continue;
    }
    if (i + 1 >= argc) throw std::invalid_argument("missing value for " + key);
    const std::string value(argv[++i]);
    const auto found = strings.find(key);
    if (found != strings.end()) *found->second = value;
    else if (key == "--core-mask") options.core_mask = parse_u32(value);
    else if (key == "--max-context-len") options.max_context_len = parse_int(value, key);
    else if (key == "--limit") options.limit = parse_int(value, key);
    else if (key == "--skip") options.skip = parse_int(value, key);
    else if (key == "--scoring-threads") options.scoring_threads = parse_int(value, key);
    else if (key == "--perf-prefill-tokens") options.perf_prefill_tokens = parse_int(value, key);
    else if (key == "--perf-decode-tokens") options.perf_decode_tokens = parse_int(value, key);
    else if (key == "--perf-warmup") options.perf_warmup = parse_int(value, key);
    else if (key == "--perf-repeat") options.perf_repeat = parse_int(value, key);
    else throw std::invalid_argument("unknown option: " + key);
  }
  if (options.model.empty() || options.weight.empty() || options.tokenizer.empty() ||
      options.embedding.empty() || options.data.empty() || options.output.empty()) {
    throw std::invalid_argument("missing required option");
  }
  if (options.model_name.empty()) options.model_name = options.model;
  if (options.adapter != "generic_text") {
    throw std::invalid_argument("unsupported adapter: " + options.adapter);
  }
  if (options.max_context_len < 2 || options.skip < 0 || options.limit < -1 ||
      options.scoring_threads < 1 || options.perf_prefill_tokens < 0 ||
      options.perf_decode_tokens < 0 || options.perf_warmup < 0 ||
      options.perf_repeat < 1) {
    throw std::invalid_argument("invalid context, skip, or limit value");
  }
  return options;
}

void check_ret(int ret, const std::string& operation) {
  if (ret < 0) throw std::runtime_error(operation + " failed: " + std::to_string(ret));
}

Runtime* initialize(const Options& options, json* runtime_info) {
  Runtime* runtime = new Runtime();
  try {
    runtime->tokenizer = new Tokenizer(TOKENIZER_BACKEND_LLAMA, options.tokenizer.c_str());
    VocabInfo vocab {};
    if (!runtime->tokenizer->GetVocabInfo(&vocab) || vocab.vocab_size <= 0) {
      throw std::runtime_error("cannot read tokenizer vocabulary");
    }
    check_ret(rknn3_init(&runtime->context, nullptr), "rknn3_init");
    check_ret(rknn3_load_model_from_path(runtime->context, options.model.c_str(),
                                         options.weight.c_str()),
              "rknn3_load_model_from_path");
    rknn3_config config {};
    config.run_core_mask = options.core_mask;
    check_ret(rknn3_model_init(runtime->context, &config), "rknn3_model_init");

    rknn3_llm_config llm_config {};
    check_ret(rknn3_query(runtime->context, RKNN3_QUERY_LLM_CONFIG, &llm_config,
                          sizeof(llm_config)),
              "RKNN3_QUERY_LLM_CONFIG");
    if (vocab.vocab_size != static_cast<int>(llm_config.vocab_size)) {
      throw std::runtime_error("tokenizer/model vocabulary mismatch");
    }
    if (options.max_context_len > static_cast<int>(llm_config.max_ctx_len)) {
      throw std::runtime_error("requested context exceeds model max_ctx_len");
    }
    runtime->embedding.open(options.embedding, vocab.vocab_size,
                            static_cast<int>(llm_config.embedding_dim));
    runtime->max_context_len = options.max_context_len;

    const uint32_t core_count = query_core_count(runtime->context);
    (*runtime_info)["model_config"] = {
        {"model_type", llm_config.model_type ? llm_config.model_type : ""},
        {"vocab_size", llm_config.vocab_size},
        {"embedding_dim", llm_config.embedding_dim},
        {"max_ctx_len", llm_config.max_ctx_len},
        {"max_position_embeddings", llm_config.max_position_embeddings},
        {"kvcache_group_size", llm_config.kvcache_group_size},
        {"core_count", core_count}};
    json attention = json::array();
    for (uint32_t i = 0; i < llm_config.n_attention_kvcache_lens; ++i) {
      json lengths = json::array();
      const auto& item = llm_config.attention_kvcache_lens[i];
      for (uint32_t j = 0; j < item.n_kvcache_buffer_lens; ++j) {
        lengths.push_back(item.kvcache_buffer_lens[j]);
      }
      attention.push_back({{"attention_type", item.attention_type},
                           {"kvcache_buffer_lengths", lengths}});
    }
    (*runtime_info)["model_config"]["attention_kvcache_lengths"] = attention;
    (*runtime_info)["memory"]["after_model_init"] = query_device_memory(runtime->context);

    rknn3_llm_param params {};
    params.logits_name = const_cast<char*>(options.logits_name.c_str());
    params.max_context_len = options.max_context_len;
    params.sampling_param.top_k = 1;
    params.sampling_param.top_p = 1.0f;
    params.sampling_param.temperature = 1.0f;
    params.sampling_param.repeat_penalty = 1.0f;
    params.vocab_info.vocab_size = vocab.vocab_size;
    params.vocab_info.n_special_bos_id = vocab.n_special_bos_id;
    params.vocab_info.n_special_eos_id = vocab.n_special_eos_id;
    params.vocab_info.linefeed_id = vocab.linefeed_id;
    params.vocab_info.ignore_eos_token = true;
    std::memcpy(params.vocab_info.special_bos_id, vocab.special_bos_id,
                sizeof(vocab.special_bos_id));
    std::memcpy(params.vocab_info.special_eos_id, vocab.special_eos_id,
                sizeof(vocab.special_eos_id));
    runtime->session = rknn3_session_init(runtime->context, &params, 1);
    if (!runtime->session) throw std::runtime_error("rknn3_session_init failed");

    runtime->scorer.vocab_size = vocab.vocab_size;
    RKLLMCallback callback {};
    callback.result_callback = result_callback;
    callback.result_userdata = runtime;
    callback.sampling_callback = sampling_callback;
    callback.sampling_userdata = &runtime->scorer;
    if (runtime->embedding.enabled()) {
      callback.embed_callback = embed_callback;
      callback.embed_userdata = &runtime->embedding;
    }
    check_ret(rknn3_session_set_callback(runtime->session, &callback),
              "rknn3_session_set_callback");

    (*runtime_info)["memory"]["after_session_init"] = query_device_memory(runtime->context);
    (*runtime_info)["memory"]["allocation"] = query_allocations(runtime->context, core_count);
    (*runtime_info)["memory"]["host"] = host_memory_json(read_host_memory());
    (*runtime_info)["context"]["kvcache_groups"] =
        query_kvcache_groups(runtime->context, core_count);
    RKLLMRunState state {};
    check_ret(rknn3_session_query_state(runtime->session, &state),
              "rknn3_session_query_state");
    (*runtime_info)["context"]["session_n_max_tokens"] = state.n_max_tokens;
    (*runtime_info)["context"]["requested_max_context_len"] = options.max_context_len;
    return runtime;
  } catch (...) {
    delete runtime;
    throw;
  }
}

json score_tokens(Runtime* runtime, const std::vector<int32_t>& prefix,
                  const std::vector<int32_t>& targets) {
  if (prefix.empty() || targets.empty()) throw std::runtime_error("empty prefix or targets");
  if (prefix.size() + targets.size() >
      static_cast<size_t>(runtime->max_context_len)) {
    throw std::runtime_error("token sequence exceeds configured Session context");
  }
  std::vector<int32_t> all_tokens(prefix);
  all_tokens.insert(all_tokens.end(), targets.begin(), targets.end());
  const std::vector<int32_t>& tokens = all_tokens;
  for (int32_t token : tokens) {
    if (token < 0 || token >= runtime->scorer.vocab_size) {
      throw std::runtime_error("token outside vocabulary");
    }
  }

  check_ret(rknn3_session_clear_kvcache(runtime->session, RKNN3_KVCACHE_CLEAR_ALL),
            "rknn3_session_clear_kvcache");
  runtime->scorer.reset(&targets, runtime->scorer.vocab_size);

  rknn3_llm_input input {};
  input.input_type = RKNN3_LLM_INPUT_TOKEN;
  input.llm_input.name = const_cast<char*>("input_ids");
  input.llm_input.tokens = const_cast<int32_t*>(prefix.data());
  input.llm_input.n_tokens = prefix.size();
  input.llm_input.enable_thinking = false;
  rknn3_llm_infer_param infer {};
  infer.keep_history = 0;
  infer.max_new_tokens = static_cast<int32_t>(targets.size());

  runtime->scorer.run_started = Clock::now();
  const int ret = rknn3_session_run(runtime->session, &input, 1, &infer);
  const auto ended = Clock::now();
  check_ret(ret, "rknn3_session_run");
  if (!runtime->scorer.error.empty()) throw std::runtime_error(runtime->scorer.error);
  if (runtime->scorer.index != targets.size()) {
    throw std::runtime_error("sampling callback count does not match target count");
  }
  RKLLMRunState state {};
  check_ret(rknn3_session_query_state(runtime->session, &state),
            "rknn3_session_query_state");

  const double elapsed = seconds_between(runtime->scorer.run_started, ended);
  const double ttft = runtime->scorer.saw_first_logits
                          ? seconds_between(runtime->scorer.run_started,
                                            runtime->scorer.first_logits)
                          : 0.0;
  const double decode_seconds = runtime->scorer.saw_first_logits
                                    ? seconds_between(runtime->scorer.first_logits, ended)
                                    : 0.0;
  const double mean_nll = runtime->scorer.nll_sum / targets.size();
  const double model_decode_seconds =
      std::max(0.0, decode_seconds - runtime->scorer.nll_compute_seconds);
  return {{"nll_sum", runtime->scorer.nll_sum},
          {"scored_tokens", targets.size()},
          {"mean_nll", mean_nll},
          {"perplexity", std::exp(mean_nll)},
          {"performance",
           {{"elapsed_seconds", elapsed},
            {"ttft_seconds", ttft},
            {"decode_seconds", decode_seconds},
            {"model_decode_seconds_estimate", model_decode_seconds},
            {"nll_compute_seconds", runtime->scorer.nll_compute_seconds},
            {"prefill_tokens", state.n_prefill_tokens},
            {"decode_tokens", state.n_decode_tokens},
            {"eval_tokens_per_second", targets.size() / elapsed},
            {"prefill_tokens_per_second", ttft > 0 ? state.n_prefill_tokens / ttft : 0.0},
            {"decode_tokens_per_second",
             decode_seconds > 0 ? state.n_decode_tokens / decode_seconds : 0.0},
            {"model_decode_tokens_per_second_estimate",
             model_decode_seconds > 0 ? state.n_decode_tokens / model_decode_seconds : 0.0},
            {"session_n_max_tokens", state.n_max_tokens}}}};
}

json score_record(Runtime* runtime, const json& record) {
  const std::vector<int32_t> tokens = record.at("tokens").get<std::vector<int32_t>>();
  const int score_from = record.at("score_from").get<int>();
  if (score_from < 1 || score_from >= static_cast<int>(tokens.size())) {
    throw std::runtime_error("invalid score_from in record " + record.at("id").get<std::string>());
  }
  std::vector<int32_t> prefix(tokens.begin(), tokens.begin() + score_from);
  std::vector<int32_t> targets(tokens.begin() + score_from, tokens.end());
  json result = score_tokens(runtime, prefix, targets);
  result["schema_version"] = 1;
  result["id"] = record.at("id");
  result["task"] = record.at("task");
  return result;
}

double percentile(std::vector<double> values, double fraction) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const size_t index = static_cast<size_t>(
      std::ceil(fraction * static_cast<double>(values.size()))) - 1;
  return values[std::min(index, values.size() - 1)];
}

json summarize_performance_samples(const json& samples) {
  const char* metrics[] = {
      "elapsed_seconds", "ttft_seconds", "prefill_tokens_per_second",
      "decode_tokens_per_second", "model_decode_tokens_per_second_estimate",
      "eval_tokens_per_second", "nll_compute_seconds"};
  json summary;
  for (const char* metric : metrics) {
    std::vector<double> values;
    for (const auto& sample : samples) {
      values.push_back(sample.at("performance").at(metric).get<double>());
    }
    summary[metric] = {{"p50", percentile(values, 0.50)},
                       {"p90", percentile(values, 0.90)}};
  }
  return summary;
}

json run_performance_probe(Runtime* runtime, const Options& options) {
  if (options.perf_prefill_tokens == 0 || options.perf_decode_tokens == 0) {
    return {{"enabled", false}};
  }
  std::ifstream data(options.data);
  std::string line;
  while (std::getline(data, line)) {
    if (line.empty()) continue;
    const json record = json::parse(line);
    const std::vector<int32_t> tokens = record.at("tokens").get<std::vector<int32_t>>();
    const size_t required = static_cast<size_t>(options.perf_prefill_tokens) +
                            static_cast<size_t>(options.perf_decode_tokens);
    if (tokens.size() < required) continue;
    std::vector<int32_t> prefix(tokens.begin(),
                                tokens.begin() + options.perf_prefill_tokens);
    std::vector<int32_t> targets(tokens.begin() + options.perf_prefill_tokens,
                                 tokens.begin() + required);
    for (int i = 0; i < options.perf_warmup; ++i) {
      score_tokens(runtime, prefix, targets);
    }
    json samples = json::array();
    for (int i = 0; i < options.perf_repeat; ++i) {
      samples.push_back(score_tokens(runtime, prefix, targets));
    }
    return {{"enabled", true},
            {"source_record_id", record.at("id")},
            {"requested_prefill_tokens", options.perf_prefill_tokens},
            {"requested_decode_tokens", options.perf_decode_tokens},
            {"warmup", options.perf_warmup},
            {"repeat", options.perf_repeat},
            {"samples", samples},
            {"percentiles", summarize_performance_samples(samples)}};
  }
  throw std::runtime_error("no record is long enough for the performance probe");
}

std::set<std::string> load_existing(const Options& options, double* nll, uint64_t* tokens) {
  std::set<std::string> completed;
  if (options.no_resume) return completed;
  std::ifstream input(options.output);
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    const json item = json::parse(line);
    const std::string id = item.at("id").get<std::string>();
    if (!completed.insert(id).second) throw std::runtime_error("duplicate output id: " + id);
    *nll += item.at("nll_sum").get<double>();
    *tokens += item.at("scored_tokens").get<uint64_t>();
  }
  return completed;
}

int run(const Options& options) {
  omp_set_dynamic(0);
  omp_set_num_threads(options.scoring_threads);
  double total_nll = 0.0;
  uint64_t total_tokens = 0;
  std::set<std::string> completed = load_existing(options, &total_nll, &total_tokens);
  std::ofstream output(options.output,
                       options.no_resume ? std::ios::trunc : std::ios::app);
  if (!output) throw std::runtime_error("cannot open output: " + options.output);

  json runtime_info;
  const auto all_started = Clock::now();
  Runtime* runtime = initialize(options, &runtime_info);
  runtime_info["scoring_threads"] = options.scoring_threads;
  std::cout << json({{"event", "runtime_ready"}, {"runtime", runtime_info}}).dump()
            << std::endl;
  int selected = 0;
  int processed = 0;
  try {
    runtime_info["performance_probe"] = run_performance_probe(runtime, options);
    std::cout << json({{"event", "performance_probe"},
                       {"result", runtime_info["performance_probe"]}}).dump()
              << std::endl;
    std::ifstream data(options.data);
    if (!data) throw std::runtime_error("cannot open data: " + options.data);
    std::string line;
    int index = 0;
    while (std::getline(data, line)) {
      if (line.empty()) continue;
      const json record = json::parse(line);
      const std::string id = record.at("id").get<std::string>();
      if (index++ < options.skip || completed.count(id)) continue;
      if (options.limit >= 0 && selected >= options.limit) break;
      ++selected;
      json result = score_record(runtime, record);
      result["model"] = options.model_name;
      result["adapter"] = options.adapter;
      total_nll += result.at("nll_sum").get<double>();
      total_tokens += result.at("scored_tokens").get<uint64_t>();
      ++processed;
      output << result.dump() << '\n';
      output.flush();
      std::cout << json({{"event", "record_done"}, {"result", result}}).dump()
                << std::endl;
    }

    runtime_info["memory"]["after_run"] = query_device_memory(runtime->context);
    runtime_info["memory"]["host_after_run"] = host_memory_json(read_host_memory());
    RKLLMRunState final_state {};
    check_ret(rknn3_session_query_state(runtime->session, &final_state),
              "rknn3_session_query_state");
    runtime_info["context"]["session_n_max_tokens_after_run"] =
        final_state.n_max_tokens;
    json summary = {{"schema_version", 1},
                    {"task", total_tokens > 0 ? "perplexity" : "benchmark"},
                    {"model", options.model_name},
                    {"adapter", options.adapter},
                    {"records_processed_this_run", processed},
                    {"records_total_in_output", completed.size() + processed},
                    {"scored_tokens", total_tokens},
                    {"elapsed_seconds", seconds_between(all_started, Clock::now())},
                    {"runtime", runtime_info}};
    summary["nll_sum"] = total_tokens > 0 ? json(total_nll) : json(nullptr);
    summary["mean_nll"] =
        total_tokens > 0 ? json(total_nll / total_tokens) : json(nullptr);
    summary["perplexity"] =
        total_tokens > 0 ? json(std::exp(total_nll / total_tokens)) : json(nullptr);
    std::ofstream summary_output(options.output + ".summary.json", std::ios::trunc);
    summary_output << std::setw(2) << summary << '\n';
    std::cout << json({{"event", "summary"}, {"summary", summary}}).dump()
              << std::endl;
  } catch (...) {
    delete runtime;
    throw;
  }
  delete runtime;
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(parse_options(argc, argv));
  } catch (const std::exception& error) {
    usage(argv[0]);
    std::cerr << "error: " << error.what() << std::endl;
    return 1;
  }
}

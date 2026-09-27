// Dump full-vocab logits from llama.cpp (libllama, tag b11211 = d7fb90e8) at chosen
// positions, for the vLLM parity check. Token ids come from bench/parity/prompts.py;
// llama.cpp never tokenizes.
//
//   llama_logits -m model.gguf -d PROMPT_DIR -o OUT_DIR [-c n_ctx] [-b n_batch]
//                [--kv f16|q8_0] [--ngl N]
//
// For each name in PROMPT_DIR/manifest.txt: reads NAME.ids and NAME.pos (int32 LE),
// decodes the sequence in n_batch chunks with outputs requested only at the listed
// positions, and writes OUT_DIR/NAME.llama.f32:
//   int32 magic 0x4C4F4731 ("LOG1"), int32 n_pos, int32 n_vocab, int32 pos[n_pos],
//   float32 logits[n_pos][n_vocab]      (row i = distribution after ids[0..pos[i]])
// The memory is cleared between sequences. One sequence at a time (n_seq_max = 1),
// flash attention on, f16 KV by default (reference quality), MTP layers not loaded.
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_set>
#include <vector>

static std::vector<int32_t> read_i32(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
    const std::streamsize n = f.tellg();
    std::vector<int32_t> v(n / sizeof(int32_t));
    f.seekg(0);
    f.read(reinterpret_cast<char *>(v.data()), n);
    return v;
}

int main(int argc, char ** argv) {
    std::string model_path, dir, out;
    int n_ctx = 0, n_batch = 2048, ngl = -1;
    std::string kv = "f16";
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(1); } return argv[++i]; };
        if (a == "-m") model_path = next();
        else if (a == "-d") dir = next();
        else if (a == "-o") out = next();
        else if (a == "-c") n_ctx = std::atoi(next().c_str());
        else if (a == "-b") n_batch = std::atoi(next().c_str());
        else if (a == "--kv") kv = next();
        else if (a == "--ngl") ngl = std::atoi(next().c_str());
        else { fprintf(stderr, "usage: %s -m model.gguf -d PROMPT_DIR -o OUT_DIR [-c n_ctx] [-b n_batch] [--kv f16|q8_0] [--ngl N]\n", argv[0]); return 1; }
    }
    if (model_path.empty() || dir.empty() || out.empty()) { fprintf(stderr, "-m, -d and -o are required\n"); return 1; }

    std::vector<std::string> names;
    {
        std::ifstream f(dir + "/manifest.txt");
        for (std::string line; std::getline(f, line);) if (!line.empty()) names.push_back(line);
    }
    if (names.empty()) { fprintf(stderr, "empty %s/manifest.txt\n", dir.c_str()); return 1; }
    if (n_ctx == 0) {
        for (auto & n : names) n_ctx = std::max<int>(n_ctx, (int) read_i32(dir + "/" + n + ".ids").size());
        n_ctx += 16;
    }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    mp.load_mtp = false;   // b11211 default; main-model logits only
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "failed to load %s\n", model_path.c_str()); return 1; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = n_batch;
    cp.n_ubatch = std::min(n_batch, 512);
    cp.n_seq_max = 1;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.no_perf = true;
    if (kv == "q8_0") { cp.type_k = GGML_TYPE_Q8_0; cp.type_v = GGML_TYPE_Q8_0; }
    else if (kv != "f16") { fprintf(stderr, "--kv f16|q8_0\n"); return 1; }
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "failed to create context (n_ctx=%d)\n", n_ctx); return 1; }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    fprintf(stderr, "n_ctx=%u n_vocab=%d kv=%s sequences=%zu\n", llama_n_ctx(ctx), n_vocab, kv.c_str(), names.size());
    llama_batch batch = llama_batch_init(n_batch, 0, 1);

    for (auto & name : names) {
        const auto ids = read_i32(dir + "/" + name + ".ids");
        const auto pos = read_i32(dir + "/" + name + ".pos");
        const std::unordered_set<int32_t> want(pos.begin(), pos.end());
        if ((int) ids.size() > (int) llama_n_ctx(ctx)) { fprintf(stderr, "%s: %zu tokens > n_ctx\n", name.c_str(), ids.size()); return 1; }
        llama_memory_clear(llama_get_memory(ctx), true);

        FILE * fo = fopen((out + "/" + name + ".llama.f32").c_str(), "wb");
        if (!fo) { fprintf(stderr, "cannot write in %s\n", out.c_str()); return 1; }
        const int32_t hdr[3] = {0x4C4F4731, (int32_t) pos.size(), n_vocab};
        fwrite(hdr, sizeof(int32_t), 3, fo);
        fwrite(pos.data(), sizeof(int32_t), pos.size(), fo);

        size_t written = 0;
        for (size_t start = 0; start < ids.size(); start += n_batch) {
            const int n = (int) std::min<size_t>(n_batch, ids.size() - start);
            batch.n_tokens = n;
            std::vector<int> outs;
            for (int j = 0; j < n; j++) {
                const int32_t p = (int32_t) (start + j);
                batch.token[j] = ids[p];
                batch.pos[j] = p;
                batch.n_seq_id[j] = 1;
                batch.seq_id[j][0] = 0;
                batch.logits[j] = want.count(p) ? 1 : 0;
                if (batch.logits[j]) outs.push_back(j);
            }
            if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "%s: llama_decode failed at %zu\n", name.c_str(), start); return 1; }
            // positions are sorted in the .pos file, and chunks are visited in order
            for (int j : outs) {
                const float * lg = llama_get_logits_ith(ctx, j);
                if (!lg) { fprintf(stderr, "%s: no logits at batch index %d\n", name.c_str(), j); return 1; }
                fwrite(lg, sizeof(float), n_vocab, fo);
                written++;
            }
        }
        fclose(fo);
        if (written != pos.size()) { fprintf(stderr, "%s: wrote %zu of %zu rows\n", name.c_str(), written, pos.size()); return 1; }
        fprintf(stderr, "%s: %zu tokens, %zu rows\n", name.c_str(), ids.size(), written);
    }
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

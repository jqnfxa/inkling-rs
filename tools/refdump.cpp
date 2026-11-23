// Reference dump: runs llama.cpp on a prompt, then greedy decode steps, and writes every named F32/I32 node.
// usage: refdump <model.gguf> <out.bin> <n_decode> <prompt text>
// build: g++ -O2 -std=c++17 refdump.cpp -I$LLAMA/include -I$LLAMA/ggml/include -L$LLAMA/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$LLAMA/build/bin -o refdump
// record: u32 step, u32 name_len, name, u32 type (0 f32, 1 i32), i64 ne[4], data (contiguous)
#include "llama.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static FILE * g_out = nullptr;
static uint32_t g_step = 0;

static bool skip_name(const char * n) {
    if (!*n || strncmp(n, "node_", 5) == 0 || strncmp(n, "leaf_", 5) == 0) return true;
    return strstr(n, "(view)") || strstr(n, "(reshaped)") || strstr(n, "(permuted)") || strstr(n, "(transposed)") || strstr(n, "(cont)") || strstr(n, "(copy");
}

static bool cb(struct ggml_tensor * t, bool ask, void *) {
    if (ask) {
        return !skip_name(t->name) && (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_I32) && ggml_nbytes(t) < (64u << 20);
    }
    std::vector<uint8_t> raw(ggml_nbytes(t));
    ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
    const int64_t n = ggml_nelements(t);
    std::vector<uint8_t> dense(n * 4);
    int64_t k = 0;
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3)
    for (int64_t i2 = 0; i2 < t->ne[2]; ++i2)
    for (int64_t i1 = 0; i1 < t->ne[1]; ++i1)
    for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
        memcpy(dense.data() + 4*k++, raw.data() + i0*t->nb[0] + i1*t->nb[1] + i2*t->nb[2] + i3*t->nb[3], 4);
    }
    const uint32_t len = strlen(t->name), type = t->type == GGML_TYPE_I32;
    fwrite(&g_step, 4, 1, g_out); fwrite(&len, 4, 1, g_out); fwrite(t->name, 1, len, g_out); fwrite(&type, 4, 1, g_out);
    fwrite(t->ne, 8, 4, g_out); fwrite(dense.data(), 1, dense.size(), g_out);
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s model out n_decode prompt\n", argv[0]); return 1; }
    const int n_decode = atoi(argv[3]);
    llama_backend_init();
    ggml_backend_load_all();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    auto cp = llama_context_default_params();
    cp.n_ctx = 4096; cp.n_batch = 512; cp.n_ubatch = 512;
    cp.cb_eval = cb; cp.n_threads = 12; cp.n_threads_batch = 12;
    llama_context * ctx = llama_init_from_model(model, cp);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    g_out = fopen(argv[2], "wb");
    std::string text = argv[4];
    std::vector<llama_token> toks(text.size() + 16);
    const int n = llama_tokenize(vocab, text.c_str(), text.size(), toks.data(), toks.size(), true, true);
    toks.resize(n);
    // step 0 records: the prompt tokens
    { const uint32_t len = 6, type = 1; int64_t ne[4] = { n, 1, 1, 1 };
      fwrite(&g_step, 4, 1, g_out); fwrite(&len, 4, 1, g_out); fwrite("tokens", 1, 6, g_out); fwrite(&type, 4, 1, g_out);
      fwrite(ne, 8, 4, g_out); fwrite(toks.data(), 4, n, g_out); }
    fprintf(stderr, "prompt tokens:"); for (auto t : toks) fprintf(stderr, " %d", t); fprintf(stderr, "\n");

    if (llama_decode(ctx, llama_batch_get_one(toks.data(), n))) return 1;
    auto * smpl = llama_sampler_init_greedy();
    for (int i = 0; i < n_decode; ++i) {
        llama_token tok = llama_sampler_sample(smpl, ctx, -1);
        g_step = i + 1;
        const uint32_t len = 5, type = 1; int64_t ne[4] = { 1, 1, 1, 1 };
        fwrite(&g_step, 4, 1, g_out); fwrite(&len, 4, 1, g_out); fwrite("input", 1, 5, g_out); fwrite(&type, 4, 1, g_out);
        fwrite(ne, 8, 4, g_out); fwrite(&tok, 4, 1, g_out);
        char buf[256]; int l = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, true);
        fprintf(stderr, "step %d token %d '%.*s'\n", i + 1, tok, l > 0 ? l : 0, buf);
        if (llama_decode(ctx, llama_batch_get_one(&tok, 1))) return 1;
    }
    fclose(g_out);
    llama_sampler_free(smpl); llama_free(ctx); llama_model_free(model);
    return 0;
}

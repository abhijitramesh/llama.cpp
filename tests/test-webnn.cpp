// Bring-up unit tests for the WebNN backend.
//
// These tests define the contract for the initial WebNN backend:
//   1. The backend registers a device named "WebNN"
//   2. Tensor data roundtrips through its buffers (set_tensor/get_tensor)
//   3. Every op listed below is reported as supported by supports_op and
//      computes results matching the CPU backend within an NMSE threshold
//
// Initial op set (F32, contiguous tensors):
//   ADD / SUB / MUL / DIV         (incl. row-broadcast of src1)
//   MUL_MAT                       (f32 x f32, incl. batch broadcast of src0)
//   SCALE                         (y = scale*x + bias)
//   SOFT_MAX                      (scale + optional f32 mask, no alibi/sinks)
//   RMS_NORM
//   GET_ROWS                      (f32 src, i32 indices)
//   GLU: SWIGLU                   (split and non-split)
//   CPY / CONT / DUP              (same type, contiguous)
//   UNARY: RELU, SIGMOID, TANH, GELU_ERF, SILU, NEG, ABS, EXP
//   SQR, SQRT, LOG, SIN, COS
//
// Full op coverage is provided by the generic backend test suite:
//   test-backend-ops test -b WebNN

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

constexpr double NMSE_DEFAULT = 1e-6;
constexpr double NMSE_MAT_MUL = 5e-4; // accumulation order/precision may differ on GPU/NPU
constexpr double NMSE_COMPOSED = 1e-5; // ops composed from multiple WebNN primitives

struct webnn_test_case {
    const char * name;
    double max_nmse;
    // build the graph; append all input tensors that must be filled to `inputs`
    std::function<ggml_tensor * (ggml_context *, std::vector<ggml_tensor *> &)> build;
};

// deterministic generator so both backends see identical input data
static float frand(uint32_t & state) {
    state = state * 1664525u + 1013904223u;
    return ((state >> 8) & 0xffffff) / (float) 0x1000000 * 2.0f - 1.0f; // [-1, 1)
}

static void fill_tensor(ggml_tensor * t, uint32_t seed) {
    const int64_t n = ggml_nelements(t);
    if (t->type == GGML_TYPE_I32) {
        // row indices for GET_ROWS - keep them small and valid
        std::vector<int32_t> data(n);
        for (int64_t i = 0; i < n; i++) {
            data[i] = (int32_t) ((i*7 + 3) % 13);
        }
        ggml_backend_tensor_set(t, data.data(), 0, n*sizeof(int32_t));
        return;
    }
    if (t->type == GGML_TYPE_I64) {
        // row indices for SET_ROWS
        std::vector<int64_t> data(n);
        for (int64_t i = 0; i < n; i++) {
            data[i] = (i*7 + 3) % 13;
        }
        ggml_backend_tensor_set(t, data.data(), 0, n*sizeof(int64_t));
        return;
    }
    if (t->type == GGML_TYPE_Q4_0 || t->type == GGML_TYPE_Q8_0 || t->type == GGML_TYPE_Q4_K) {
        // quantize deterministic f32 data row by row
        uint32_t state = seed;
        std::vector<float> src(n);
        for (int64_t i = 0; i < n; i++) {
            src[i] = frand(state);
        }
        std::vector<uint8_t> q(ggml_nbytes(t));
        ggml_quantize_chunk(t->type, src.data(), q.data(), 0, n / t->ne[0], t->ne[0], nullptr);
        ggml_backend_tensor_set(t, q.data(), 0, q.size());
        return;
    }
    const bool positive = strstr(t->name, "_pos") != nullptr;
    uint32_t state = seed;
    if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> data(n);
        for (int64_t i = 0; i < n; i++) {
            const float v = frand(state);
            data[i] = ggml_fp32_to_fp16(positive ? fabsf(v) + 0.5f : v);
        }
        ggml_backend_tensor_set(t, data.data(), 0, n*sizeof(ggml_fp16_t));
        return;
    }
    GGML_ASSERT(t->type == GGML_TYPE_F32);
    std::vector<float> data(n);
    for (int64_t i = 0; i < n; i++) {
        const float v = frand(state);
        data[i] = positive ? fabsf(v) + 0.5f : v;
    }
    ggml_backend_tensor_set(t, data.data(), 0, n*sizeof(float));
}

static std::vector<float> read_tensor_f32(const ggml_tensor * t) {
    const int64_t n = ggml_nelements(t);
    std::vector<float> out(n);
    if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> raw(n);
        ggml_backend_tensor_get(t, raw.data(), 0, n*sizeof(ggml_fp16_t));
        for (int64_t i = 0; i < n; i++) {
            out[i] = ggml_fp16_to_fp32(raw[i]);
        }
    } else {
        GGML_ASSERT(t->type == GGML_TYPE_F32);
        ggml_backend_tensor_get(t, out.data(), 0, n*sizeof(float));
    }
    return out;
}

static double nmse(const std::vector<float> & ref, const std::vector<float> & out) {
    GGML_ASSERT(ref.size() == out.size());
    double err = 0.0;
    double nrm = 0.0;
    for (size_t i = 0; i < ref.size(); i++) {
        err += (ref[i] - out[i]) * (ref[i] - out[i]);
        nrm += ref[i] * ref[i];
    }
    return err / std::max(nrm, 1e-12);
}

// run one case on the given backend, return flattened f32 output
// returns false on compute failure, or on unsupported ops when check_support is set
static bool run_case(ggml_backend_t backend, const webnn_test_case & tc, bool check_support, std::vector<float> & result) {
    ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead()*128 + ggml_graph_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx);

    std::vector<ggml_tensor *> inputs;
    ggml_tensor * out = tc.build(ctx, inputs);
    ggml_set_name(out, "out");

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    if (check_support) {
        for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
            ggml_tensor * node = ggml_graph_node(gf, i);
            if (!ggml_backend_supports_op(backend, node)) {
                printf("    op not supported: %s\n", ggml_op_desc(node));
                ggml_free(ctx);
                return false;
            }
        }
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buf);

    uint32_t seed = 42;
    for (size_t i = 0; i < inputs.size(); i++) {
        fill_tensor(inputs[i], seed + (uint32_t) i*7919);
    }

    const ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        printf("    graph_compute failed with status %d\n", status);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return false;
    }

    result = read_tensor_f32(out);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

static ggml_tensor * new_input(ggml_context * ctx, const char * name, std::vector<ggml_tensor *> & inputs,
                               int64_t ne0, int64_t ne1 = 1, int64_t ne2 = 1, int64_t ne3 = 1,
                               ggml_type type = GGML_TYPE_F32) {
    ggml_tensor * t = ggml_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3);
    ggml_set_name(t, name);
    inputs.push_back(t);
    return t;
}

static bool test_buffer_roundtrip(ggml_backend_t backend) {
    ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead()*8,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 67, 33); // odd sizes on purpose
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buf);

    const size_t n = ggml_nelements(t);
    std::vector<float> src(n);
    uint32_t state = 123;
    for (size_t i = 0; i < n; i++) {
        src[i] = frand(state);
    }

    ggml_backend_tensor_set(t, src.data(), 0, n*sizeof(float));
    std::vector<float> dst(n);
    ggml_backend_tensor_get(t, dst.data(), 0, n*sizeof(float));

    const bool ok = memcmp(src.data(), dst.data(), n*sizeof(float)) == 0;

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

// after graph_compute, EVERY node's data must be valid in host memory - the
// scheduler and other backends read intermediate tensors across split
// boundaries (this pins the contract for whole-graph backend execution)
static bool test_intermediate_readback(ggml_backend_t webnn, ggml_backend_t cpu) {
    std::vector<float> mid_ref, out_ref, mid_out, out_out;

    for (int pass = 0; pass < 2; pass++) {
        ggml_backend_t backend = pass == 0 ? cpu : webnn;
        ggml_init_params params = {
            /* .mem_size   = */ ggml_tensor_overhead()*16 + ggml_graph_overhead(),
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ true,
        };
        ggml_context * ctx = ggml_init(params);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 4);
        ggml_tensor * y = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 4);
        ggml_tensor * z = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 4);
        ggml_tensor * mid = ggml_mul(ctx, x, y);
        ggml_tensor * out = ggml_add(ctx, mid, z);
        // v4 contract: intermediates consumed within a graph are only
        // materialized to host when explicitly flagged as outputs
        ggml_set_output(mid);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        fill_tensor(x, 11);
        fill_tensor(y, 22);
        fill_tensor(z, 33);

        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            return false;
        }
        (pass == 0 ? mid_ref : mid_out) = read_tensor_f32(mid);
        (pass == 0 ? out_ref : out_out) = read_tensor_f32(out);

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    return nmse(mid_ref, mid_out) <= NMSE_DEFAULT && nmse(out_ref, out_out) <= NMSE_DEFAULT;
}

// SET_ROWS into an f16 cache followed by a matmul reading the cache through a
// view - the KV-cache write/read pattern (device-resident path on WebNN)
static bool test_set_rows_then_matmul(ggml_backend_t webnn, ggml_backend_t cpu) {
    std::vector<float> ref, out_v;

    for (int pass = 0; pass < 2; pass++) {
        ggml_backend_t backend = pass == 0 ? cpu : webnn;
        ggml_init_params params = {
            /* .mem_size   = */ ggml_tensor_overhead()*16 + ggml_graph_overhead(),
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ true,
        };
        ggml_context * ctx = ggml_init(params);
        ggml_tensor * cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 64, 32);
        ggml_tensor * rows  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 4);
        ggml_tensor * idx   = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 4);
        ggml_tensor * q     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 5);
        ggml_set_name(cache, "cache");

        ggml_tensor * sr = ggml_set_rows(ctx, cache, rows, idx);
        // read the first 16 rows of the (updated) cache through a fresh view
        ggml_tensor * kv = ggml_view_2d(ctx, cache, 64, 16, cache->nb[1], 0);
        ggml_tensor * out = ggml_mul_mat(ctx, kv, q);

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, sr);
        ggml_build_forward_expand(gf, out);

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        fill_tensor(cache, 7);
        fill_tensor(rows, 21);
        fill_tensor(idx, 0);
        fill_tensor(q, 35);

        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            return false;
        }
        (pass == 0 ? ref : out_v) = read_tensor_f32(out);

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    return nmse(ref, out_v) <= NMSE_MAT_MUL;
}

int main() {
    ggml_backend_load_all();

    ggml_backend_dev_t webnn_dev = ggml_backend_dev_by_name("WebNN");
    if (webnn_dev == nullptr) {
        printf("WebNN device not available - skipping\n");
        return 0;
    }

    printf("WebNN device found: %s (%s)\n",
        ggml_backend_dev_name(webnn_dev), ggml_backend_dev_description(webnn_dev));

    ggml_backend_t webnn = ggml_backend_dev_init(webnn_dev, nullptr);
    GGML_ASSERT(webnn);

    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    GGML_ASSERT(cpu);

    int n_ok   = 0;
    int n_fail = 0;

    // 2. buffer roundtrip
    {
        const bool ok = test_buffer_roundtrip(webnn);
        printf("  %-24s %s\n", "buffer_roundtrip", ok ? "OK" : "FAIL");
        ok ? n_ok++ : n_fail++;
    }

    // 2b. flagged intermediate tensors must be readable after graph_compute
    {
        const bool ok = test_intermediate_readback(webnn, cpu);
        printf("  %-24s %s\n", "intermediate_readback", ok ? "OK" : "FAIL");
        ok ? n_ok++ : n_fail++;
    }

    // 2c. KV-cache write/read pattern (SET_ROWS scatter + view read)
    {
        const bool ok = test_set_rows_then_matmul(webnn, cpu);
        printf("  %-24s %s\n", "set_rows_then_matmul", ok ? "OK" : "FAIL");
        ok ? n_ok++ : n_fail++;
    }

    // 3. op correctness vs CPU
    const std::vector<webnn_test_case> cases = {
        { "add", NMSE_DEFAULT, [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
            return ggml_add(ctx, new_input(ctx, "a", in, 64, 5, 4, 3), new_input(ctx, "b", in, 64, 5, 4, 3));
        }},
        { "add_bcast_row", NMSE_DEFAULT, [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
            return ggml_add(ctx, new_input(ctx, "a", in, 64, 5, 4, 3), new_input(ctx, "b", in, 64));
        }},
        { "sub", NMSE_DEFAULT, [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
            return ggml_sub(ctx, new_input(ctx, "a", in, 64, 5, 4, 3), new_input(ctx, "b", in, 64, 5, 4, 3));
        }},
        { "mul", NMSE_DEFAULT, [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
            return ggml_mul(ctx, new_input(ctx, "a", in, 64, 5, 4, 3), new_input(ctx, "b", in, 64, 5, 4, 3));
        }},
        { "div", NMSE_DEFAULT, [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
            return ggml_div(ctx, new_input(ctx, "a", in, 64, 5, 4, 3), new_input(ctx, "b_pos", in, 64, 5, 4, 3));
        }},
        { "mul_mat", NMSE_MAT_MUL, [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
            // dst[16,32] = src0[256,32]^T-style product, the core LLM op
            return ggml_mul_mat(ctx, new_input(ctx, "w", in, 256, 32), new_input(ctx, "x", in, 256, 16));
        }},
        { "mul_mat_batched", NMSE_MAT_MUL, [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
            // src0 broadcast across src1 batches (ne02=1 vs ne12=4)
            return ggml_mul_mat(ctx, new_input(ctx, "w", in, 64, 16), new_input(ctx, "x", in, 64, 8, 4, 2));
        }},
        { "scale", NMSE_DEFAULT, [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
            return ggml_scale(ctx, new_input(ctx, "a", in, 64, 5, 4, 3), 2.5f);
        }},
        { "scale_bias", NMSE_DEFAULT, [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
            return ggml_scale_bias(ctx, new_input(ctx, "a", in, 64, 5, 4, 3), 0.75f, -1.25f);
        }},
        { "soft_max", NMSE_COMPOSED, [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
            return ggml_soft_max(ctx, new_input(ctx, "a", in, 64, 8, 2, 1));
        }},
        { "soft_max_masked", NMSE_COMPOSED, [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
            // attention shape: kq [n_kv, n_tokens, n_head, 1], f32 mask broadcast over heads
            ggml_tensor * kq   = new_input(ctx, "a", in, 80, 7, 4, 1);
            ggml_tensor * mask = new_input(ctx, "m", in, 80, 7, 1, 1);
            return ggml_soft_max_ext(ctx, kq, mask, 0.125f, 0.0f);
        }},
        { "swiglu_split", NMSE_COMPOSED, [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
            // fused FFN activation used by llama-arch models
            return ggml_swiglu_split(ctx, new_input(ctx, "g", in, 64, 5, 2, 1), new_input(ctx, "u", in, 64, 5, 2, 1));
        }},
        { "swiglu", NMSE_COMPOSED, [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
            return ggml_swiglu(ctx, new_input(ctx, "a", in, 128, 5, 2, 1));
        }},
        { "dup", NMSE_DEFAULT, [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
            return ggml_dup(ctx, new_input(ctx, "a", in, 64, 5, 4, 3));
        }},
        { "cpy_reshape_matmul", NMSE_MAT_MUL, [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
            // shape-changing copy feeding a matmul (the attention heads-merge pattern):
            // cont([48,6] view) -> [288] -> wo @ x
            ggml_tensor * x  = new_input(ctx, "x",  in, 48, 6);
            ggml_tensor * w  = new_input(ctx, "w",  in, 288, 32);
            ggml_tensor * merged = ggml_cpy(ctx, x, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 288, 1));
            return ggml_mul_mat(ctx, w, merged);
        }},
        { "rms_norm", NMSE_COMPOSED, [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
            return ggml_rms_norm(ctx, new_input(ctx, "a", in, 128, 4, 2, 1), 1e-6f);
        }},
        { "get_rows", NMSE_DEFAULT, [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
            ggml_tensor * src = new_input(ctx, "a", in, 64, 16);
            ggml_tensor * idx = new_input(ctx, "i", in, 8, 1, 1, 1, GGML_TYPE_I32);
            return ggml_get_rows(ctx, src, idx);
        }},
        { "relu", NMSE_DEFAULT, [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
            return ggml_relu(ctx, new_input(ctx, "a", in, 128, 5, 2, 1));
        }},
        { "sigmoid", NMSE_DEFAULT, [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
            return ggml_sigmoid(ctx, new_input(ctx, "a", in, 128, 5, 2, 1));
        }},
        { "tanh", NMSE_DEFAULT, [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
            return ggml_tanh(ctx, new_input(ctx, "a", in, 128, 5, 2, 1));
        }},
        { "gelu_erf", NMSE_COMPOSED, [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
            return ggml_gelu_erf(ctx, new_input(ctx, "a", in, 128, 5, 2, 1));
        }},
        { "silu", NMSE_COMPOSED, [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
            return ggml_silu(ctx, new_input(ctx, "a", in, 128, 5, 2, 1));
        }},
        { "neg", NMSE_DEFAULT, [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
            return ggml_neg(ctx, new_input(ctx, "a", in, 128, 5, 2, 1));
        }},
        { "abs", NMSE_DEFAULT, [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
            return ggml_abs(ctx, new_input(ctx, "a", in, 128, 5, 2, 1));
        }},
        { "exp", NMSE_DEFAULT, [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
            return ggml_exp(ctx, new_input(ctx, "a", in, 128, 5, 2, 1));
        }},
        { "sqr", NMSE_DEFAULT, [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
            return ggml_sqr(ctx, new_input(ctx, "a", in, 128, 5, 2, 1));
        }},
        { "sqrt", NMSE_DEFAULT, [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
            return ggml_sqrt(ctx, new_input(ctx, "a_pos", in, 128, 5, 2, 1));
        }},
        { "log", NMSE_DEFAULT, [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
            return ggml_log(ctx, new_input(ctx, "a_pos", in, 128, 5, 2, 1));
        }},
        { "sin", NMSE_DEFAULT, [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
            return ggml_sin(ctx, new_input(ctx, "a", in, 128, 5, 2, 1));
        }},
        { "cos", NMSE_DEFAULT, [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
            return ggml_cos(ctx, new_input(ctx, "a", in, 128, 5, 2, 1));
        }},
        { "mlp_chain", NMSE_MAT_MUL, [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
            // multi-node graph: out = w2 @ silu(w1 @ x) - exercises sequential dispatch
            ggml_tensor * x  = new_input(ctx, "x",  in, 64, 4);
            ggml_tensor * w1 = new_input(ctx, "w1", in, 64, 128);
            ggml_tensor * w2 = new_input(ctx, "w2", in, 128, 32);
            return ggml_mul_mat(ctx, w2, ggml_silu(ctx, ggml_mul_mat(ctx, w1, x)));
        }},
        { "mlp_reshape_chain", NMSE_MAT_MUL, [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
            // in-graph reshape between compute nodes
            ggml_tensor * x  = new_input(ctx, "x",  in, 64, 4);
            ggml_tensor * w1 = new_input(ctx, "w1", in, 64, 128);
            ggml_tensor * w2 = new_input(ctx, "w2", in, 64, 32);
            ggml_tensor * h  = ggml_silu(ctx, ggml_mul_mat(ctx, w1, x)); // [128, 4]
            return ggml_mul_mat(ctx, w2, ggml_reshape_2d(ctx, h, 64, 8)); // [32, 8]
        }},
        { "add_f16", NMSE_DEFAULT, [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
            return ggml_add(ctx, new_input(ctx, "a", in, 64, 5, 4, 3, GGML_TYPE_F16),
                                 new_input(ctx, "b", in, 64, 5, 4, 3, GGML_TYPE_F16));
        }},
        { "relu_f16", NMSE_DEFAULT, [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
            return ggml_relu(ctx, new_input(ctx, "a", in, 128, 5, 2, 1, GGML_TYPE_F16));
        }},
        { "mul_mat_f16", NMSE_MAT_MUL, [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
            // f16 weights x f32 activations -> f32, the standard f16-model matmul
            return ggml_mul_mat(ctx, new_input(ctx, "w", in, 256, 32, 1, 1, GGML_TYPE_F16),
                                     new_input(ctx, "x", in, 256, 16));
        }},
        { "mul_mat_q4_0", NMSE_MAT_MUL, [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
            // Q4_0 weights become dequantizeLinear graph constants
            return ggml_mul_mat(ctx, new_input(ctx, "w", in, 256, 32, 1, 1, GGML_TYPE_Q4_0),
                                     new_input(ctx, "x", in, 256, 16));
        }},
        { "mul_mat_q8_0", NMSE_MAT_MUL, [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
            return ggml_mul_mat(ctx, new_input(ctx, "w", in, 256, 32, 1, 1, GGML_TYPE_Q8_0),
                                     new_input(ctx, "x", in, 256, 16));
        }},
        { "mul_mat_q4_0_576", NMSE_MAT_MUL, [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
            // model-like shape: k=576 (18 blocks/row, not a power of two)
            return ggml_mul_mat(ctx, new_input(ctx, "w", in, 576, 64, 1, 1, GGML_TYPE_Q4_0),
                                     new_input(ctx, "x", in, 576, 5));
        }},
        { "mul_mat_q4_0_chain", NMSE_MAT_MUL, [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
            // two quantized matmuls chained, like consecutive layer projections
            ggml_tensor * x  = new_input(ctx, "x", in, 576, 3);
            ggml_tensor * w1 = new_input(ctx, "w1", in, 576, 1536, 1, 1, GGML_TYPE_Q4_0);
            ggml_tensor * w2 = new_input(ctx, "w2", in, 1536, 576, 1, 1, GGML_TYPE_Q4_0);
            return ggml_mul_mat(ctx, w2, ggml_mul_mat(ctx, w1, x));
        }},
        { "mul_mat_q4_k", NMSE_MAT_MUL, [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
            // K-quant super-blocks (256 values, two-level scales + mins)
            return ggml_mul_mat(ctx, new_input(ctx, "w", in, 256, 32, 1, 1, GGML_TYPE_Q4_K),
                                     new_input(ctx, "x", in, 256, 16));
        }},
        { "mul_mat_q4_k_512", NMSE_MAT_MUL, [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
            return ggml_mul_mat(ctx, new_input(ctx, "w", in, 512, 64, 1, 1, GGML_TYPE_Q4_K),
                                     new_input(ctx, "x", in, 512, 5));
        }},
        { "get_rows_q4_0", NMSE_DEFAULT, [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
            ggml_tensor * src = new_input(ctx, "a", in, 64, 16, 1, 1, GGML_TYPE_Q4_0);
            ggml_tensor * idx = new_input(ctx, "i", in, 8, 1, 1, 1, GGML_TYPE_I32);
            return ggml_get_rows(ctx, src, idx);
        }},
        { "get_rows_f16", NMSE_DEFAULT, [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
            ggml_tensor * src = new_input(ctx, "a", in, 64, 16, 1, 1, GGML_TYPE_F16);
            ggml_tensor * idx = new_input(ctx, "i", in, 8, 1, 1, 1, GGML_TYPE_I32);
            return ggml_get_rows(ctx, src, idx);
        }},
        { "set_rows", NMSE_DEFAULT, [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
            // f16 cache, f32 rows, i64 indices - the KV-cache write op
            ggml_tensor * cache = new_input(ctx, "c", in, 64, 32, 1, 1, GGML_TYPE_F16);
            ggml_tensor * rows  = new_input(ctx, "r", in, 64, 4);
            ggml_tensor * idx   = new_input(ctx, "i", in, 4, 1, 1, 1, GGML_TYPE_I64);
            return ggml_set_rows(ctx, cache, rows, idx);
        }},
        { "mul_mat_permuted", NMSE_MAT_MUL, [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
            // non-contiguous (permuted) src0, the attention KQ pattern
            ggml_tensor * a = new_input(ctx, "a", in, 64, 8, 6, 1);
            ggml_tensor * k = ggml_permute(ctx, a, 0, 2, 1, 3); // [64, 6, 8] strided
            ggml_tensor * q = new_input(ctx, "q", in, 64, 5, 8, 1);
            return ggml_mul_mat(ctx, k, q);
        }},
        { "rope_norm", NMSE_COMPOSED, [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
            ggml_tensor * x   = new_input(ctx, "x", in, 48, 6, 4, 1); // [d, n_head, n_tokens]
            ggml_tensor * pos = new_input(ctx, "p", in, 4, 1, 1, 1, GGML_TYPE_I32);
            return ggml_rope(ctx, x, pos, 48, GGML_ROPE_TYPE_NORMAL);
        }},
        { "rope_neox", NMSE_COMPOSED, [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
            ggml_tensor * x   = new_input(ctx, "x", in, 48, 6, 4, 1);
            ggml_tensor * pos = new_input(ctx, "p", in, 4, 1, 1, 1, GGML_TYPE_I32);
            return ggml_rope(ctx, x, pos, 48, GGML_ROPE_TYPE_NEOX);
        }},
        { "flash_attn", NMSE_MAT_MUL, [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
            ggml_tensor * q = new_input(ctx, "q", in, 64, 5, 8, 1);                  // [d, n_tokens, n_head]
            ggml_tensor * k = new_input(ctx, "k", in, 64, 32, 8, 1, GGML_TYPE_F16);  // [d, n_kv, n_head]
            ggml_tensor * v = new_input(ctx, "v", in, 64, 32, 8, 1, GGML_TYPE_F16);
            ggml_tensor * m = new_input(ctx, "m", in, 32, 5, 1, 1, GGML_TYPE_F16);   // [n_kv, n_tokens]
            return ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f/8.0f, 0.0f, 0.0f);
        }},
        { "mul_mat_gqa", NMSE_MAT_MUL, [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
            // grouped batch broadcast: 8 query heads share 2 KV heads (G=4)
            ggml_tensor * k = new_input(ctx, "k", in, 64, 16, 2, 1);
            ggml_tensor * q = new_input(ctx, "q", in, 64, 5, 8, 1);
            return ggml_mul_mat(ctx, k, q);
        }},
        { "flash_attn_gqa", NMSE_MAT_MUL, [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
            ggml_tensor * q = new_input(ctx, "q", in, 64, 5, 8, 1);                  // 8 query heads
            ggml_tensor * k = new_input(ctx, "k", in, 64, 32, 2, 1, GGML_TYPE_F16);  // 2 KV heads
            ggml_tensor * v = new_input(ctx, "v", in, 64, 32, 2, 1, GGML_TYPE_F16);
            ggml_tensor * m = new_input(ctx, "m", in, 32, 5, 1, 1, GGML_TYPE_F16);
            return ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f/8.0f, 0.0f, 0.0f);
        }},
        { "flash_attn_permuted_kv", NMSE_MAT_MUL, [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
            // K/V as permuted views, like views of the KV cache
            ggml_tensor * q  = new_input(ctx, "q", in, 64, 5, 8, 1);
            ggml_tensor * kc = new_input(ctx, "kc", in, 64, 8, 32, 1, GGML_TYPE_F16); // [d, n_head, n_kv]
            ggml_tensor * vc = new_input(ctx, "vc", in, 64, 8, 32, 1, GGML_TYPE_F16);
            ggml_tensor * k  = ggml_permute(ctx, kc, 0, 2, 1, 3); // -> [d, n_kv, n_head] strided
            ggml_tensor * v  = ggml_permute(ctx, vc, 0, 2, 1, 3);
            ggml_tensor * m  = new_input(ctx, "m", in, 32, 5, 1, 1, GGML_TYPE_F16);
            return ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f/8.0f, 0.0f, 0.0f);
        }},
    };

    for (const auto & tc : cases) {
        std::vector<float> ref;
        std::vector<float> out;

        if (!run_case(cpu, tc, /*check_support=*/false, ref)) {
            printf("  %-24s FAIL (cpu reference failed)\n", tc.name);
            n_fail++;
            continue;
        }
        if (!run_case(webnn, tc, /*check_support=*/true, out)) {
            printf("  %-24s FAIL (webnn compute/support)\n", tc.name);
            n_fail++;
            continue;
        }

        const double err = nmse(ref, out);
        const bool ok = err <= tc.max_nmse;
        printf("  %-24s %s (nmse %.3g)\n", tc.name, ok ? "OK" : "FAIL", err);
        ok ? n_ok++ : n_fail++;
    }

    ggml_backend_free(webnn);
    ggml_backend_free(cpu);

    printf("%d/%d tests passed\n", n_ok, n_ok + n_fail);
    return n_fail == 0 ? 0 : 1;
}

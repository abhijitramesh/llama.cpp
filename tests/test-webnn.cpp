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
//   SOFT_MAX                      (no mask, scale only)
//   RMS_NORM
//   GET_ROWS                      (f32 src, i32 indices)
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
    GGML_ASSERT(t->type == GGML_TYPE_F32);
    const bool positive = strstr(t->name, "_pos") != nullptr;
    uint32_t state = seed;
    std::vector<float> data(n);
    for (int64_t i = 0; i < n; i++) {
        const float v = frand(state);
        data[i] = positive ? fabsf(v) + 0.5f : v;
    }
    ggml_backend_tensor_set(t, data.data(), 0, n*sizeof(float));
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

    GGML_ASSERT(out->type == GGML_TYPE_F32);
    result.resize(ggml_nelements(out));
    ggml_backend_tensor_get(out, result.data(), 0, result.size()*sizeof(float));

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

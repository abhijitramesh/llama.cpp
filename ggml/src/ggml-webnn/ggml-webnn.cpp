// WebNN backend for ggml
//
// WebNN (navigator.ml) is a graph-based browser API: ops are recorded into an
// MLGraphBuilder, compiled once with build(), and then executed repeatedly with
// dispatch() on MLTensor handles. There is no native C API, so this backend is
// Emscripten-only and talks to the browser through EM_ASYNC_JS bindings (which
// require JSPI or ASYNCIFY to suspend the wasm stack across JS promises).
//
// Initial design (correctness over speed):
//  - tensor data lives in host (wasm heap) memory, reusing the CPU buffer type
//  - graph_compute dispatches each ggml node as a single-op WebNN graph
//  - compiled MLGraphs and their MLTensor handles are cached per op signature
//    (op + shapes + params), so steady-state cost is write/dispatch/read
//
// Future work: device-resident MLTensors, whole-cgraph compilation, f16/quants.

#include "ggml-webnn.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <emscripten/emscripten.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#ifdef GGML_WEBNN_DEBUG
#    define WEBNN_LOG_DEBUG(...) GGML_LOG_DEBUG(__VA_ARGS__)
#else
#    define WEBNN_LOG_DEBUG(...) ((void) 0)
#endif

//
// JS bindings
//

// create the MLContext and install the graph cache + builder on globalThis.
// returns 1 on success, 0 when WebNN is unavailable.
EM_ASYNC_JS(int, ggml_webnn_js_init, (), {
    if (typeof navigator === 'undefined' || !navigator.ml) {
        return 0;
    }
    try {
        /* device preference (cpu/gpu/npu) can be set by the embedding page via
           globalThis.GGML_WEBNN_DEVICE; falls back to the UA default context */
        let context;
        const pref = globalThis.GGML_WEBNN_DEVICE;
        if (pref) {
            try {
                context = await navigator.ml.createContext({ deviceType : pref });
                console.log('ggml-webnn: created MLContext with deviceType=' + pref);
            } catch (e) {
                console.warn('ggml-webnn: deviceType=' + pref + ' rejected, using default context:', e);
            }
        }
        if (!context) {
            context = await navigator.ml.createContext();
        }
        const S = {
            context : context,
            graphs  : new Map(),
        };
        /* build (or fetch from cache) the single-op graph described by descStr.
           desc format: {op, in: [jsShape, ...], dt: [dataType, ...]?, out: jsShape, ...params}
           input i of the graph is named 'i' + i; the single output is named 'out'. */
        S.getEntry = async function(descStr) {
            let entry = S.graphs.get(descStr);
            if (entry) {
                return entry;
            }
            const d = JSON.parse(descStr);
            const b = new MLGraphBuilder(S.context);
            const dts = d.dt || [];
            const ins = [];
            const x = [];
            for (let i = 0; i < d.in.length; i++) {
                const dataType = dts[i] || 'float32';
                const shape = d.in[i];
                ins.push({ name : 'i' + i, shape : shape, dataType : dataType });
                x.push(b.input('i' + i, { dataType : dataType, shape : shape, dimensions : shape }));
            }
            let out;
            switch (d.op) {
                case 'add':
                case 'sub':
                case 'mul':
                case 'div':
                    out = b[d.op](x[0], x[1]);
                    break;
                case 'matmul':
                {
                    /* ggml mul_mat: dst[n,m] = dot(src0 row m, src1 row n)
                       JS shapes are [b3, b2, rows, cols], so: matmul(src1, src0^T) */
                    out = b.matmul(x[1], b.transpose(x[0], { permutation : [0, 1, 3, 2] }));
                    break;
                }
                case 'unary':
                    out = b[d.fn](x[0]);
                    break;
                case 'silu':
                    out = b.mul(x[0], b.sigmoid(x[0]));
                    break;
                case 'sqr':
                    out = b.mul(x[0], x[0]);
                    break;
                case 'linear':
                    out = b.linear(x[0], { alpha : d.alpha, beta : d.beta });
                    break;
                case 'softmax':
                {
                    let t = x[0];
                    if (d.scale !== 1) {
                        t = b.linear(t, { alpha : d.scale, beta : 0 });
                    }
                    if (x.length > 1) {
                        t = b.add(t, x[1]); /* additive attention mask, broadcast over heads */
                    }
                    out = b.softmax(t, d.in[0].length - 1);
                    break;
                }
                case 'swiglu_split':
                {
                    const g = x[d.gate];
                    const u = x[1 - d.gate];
                    out = b.mul(b.mul(g, b.sigmoid(g)), u);
                    break;
                }
                case 'swiglu':
                {
                    const s = d.in[0];
                    const n = s[3] / 2;
                    const sizes = [s[0], s[1], s[2], n];
                    const g = b.slice(x[0], [0, 0, 0, d.swapped ? n : 0], sizes);
                    const u = b.slice(x[0], [0, 0, 0, d.swapped ? 0 : n], sizes);
                    out = b.mul(b.mul(g, b.sigmoid(g)), u);
                    break;
                }
                case 'rms_norm':
                {
                    const ax = d.in[0].length - 1;
                    const ms = b.reduceMean(b.mul(x[0], x[0]), { axes : [ax], keepDimensions : true });
                    const eps = b.constant({ dataType : 'float32', shape : [1], dimensions : [1] },
                                           new Float32Array([d.eps]));
                    out = b.div(x[0], b.sqrt(b.add(ms, eps)));
                    break;
                }
                case 'get_rows':
                    out = b.gather(x[0], x[1], { axis : 0 });
                    break;
                default:
                    throw new Error('unknown op ' + d.op);
            }
            const graph = await b.build({ out : out });
            const tensors = [];
            for (const i of ins) {
                tensors.push(await S.context.createTensor(
                    { dataType : i.dataType, shape : i.shape, dimensions : i.shape, writable : true }));
            }
            const outTensor = await S.context.createTensor(
                { dataType : 'float32', shape : d.out, dimensions : d.out, readable : true });
            entry = { graph : graph, inputs : tensors, outTensor : outTensor };
            S.graphs.set(descStr, entry);
            return entry;
        };
        globalThis.__ggml_webnn = S;
        return 1;
    } catch (e) {
        console.warn('ggml-webnn: initialization failed:', e);
        return 0;
    }
});

// execute one op: copy inputs from the wasm heap into MLTensors, dispatch the
// cached graph, read the result back into the wasm heap. returns 0 on success.
EM_ASYNC_JS(int, ggml_webnn_js_dispatch,
            (const char * desc, void * p0, size_t n0, void * p1, size_t n1, void * pdst, size_t ndst), {
    const S = globalThis.__ggml_webnn;
    try {
        /* under JSPI the wasm ABI passes these as BigInt - coerce before any arithmetic */
        const entry = await S.getEntry(UTF8ToString(Number(desc)));
        const ptrs = [Number(p0), Number(p1)];
        const lens = [Number(n0), Number(n1)];
        pdst = Number(pdst);
        ndst = Number(ndst);
        const feeds = {};
        for (let i = 0; i < entry.inputs.length; i++) {
            /* writeTensor copies synchronously, so a heap view is safe here */
            S.context.writeTensor(entry.inputs[i], HEAPU8.subarray(ptrs[i], ptrs[i] + lens[i]));
            feeds['i' + i] = entry.inputs[i];
        }
        S.context.dispatch(entry.graph, feeds, { out : entry.outTensor });
        const buf = await S.context.readTensor(entry.outTensor);
        if (buf.byteLength !== ndst) {
            throw new Error('output size mismatch: ' + buf.byteLength + ' != ' + ndst);
        }
        /* HEAPU8 is re-read here, after the await - memory may have grown */
        HEAPU8.set(new Uint8Array(buf), pdst);
        return 0;
    } catch (e) {
        console.error('ggml-webnn: dispatch failed:', e);
        return 1;
    }
});

static bool   g_webnn_available  = false;
static size_t g_webnn_n_dispatch = 0; // ops executed on WebNN, reported on backend free

//
// op -> graph descriptor encoding
//

// JS (row-major) shape of a ggml tensor: reversed ne, e.g. rank 4 -> [ne3,ne2,ne1,ne0]
static std::string ggml_webnn_shape_js(const ggml_tensor * t, int rank = 4) {
    std::string s = "[";
    for (int i = rank - 1; i >= 0; i--) {
        s += std::to_string(t->ne[i]);
        if (i > 0) {
            s += ",";
        }
    }
    s += "]";
    return s;
}

static std::string ggml_webnn_float_js(float v) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%.9g", v);
    return buf;
}

// builder method name for the supported subset of unary ops, NULL otherwise
static const char * ggml_webnn_unary_fn(const ggml_tensor * op) {
    switch (ggml_get_unary_op(op)) {
        case GGML_UNARY_OP_RELU:     return "relu";
        case GGML_UNARY_OP_SIGMOID:  return "sigmoid";
        case GGML_UNARY_OP_TANH:     return "tanh";
        case GGML_UNARY_OP_GELU_ERF: return "gelu"; // WebNN gelu is the erf formulation
        case GGML_UNARY_OP_NEG:      return "neg";
        case GGML_UNARY_OP_ABS:      return "abs";
        case GGML_UNARY_OP_EXP:      return "exp";
        default:                     return nullptr;
    }
}

// encode a node into a graph descriptor; returns false for unhandled ops.
// the descriptor doubles as the graph cache key, so it must capture every
// value that changes the compiled graph (op, shapes, dtypes, params).
static bool ggml_webnn_encode_desc(const ggml_tensor * node, std::string & desc, int & n_inputs) {
    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];

    n_inputs = src1 ? 2 : 1;

    auto desc_simple = [&](const std::string & op_fields) {
        desc = "{" + op_fields + ",\"in\":[" + ggml_webnn_shape_js(src0);
        if (src1) {
            desc += "," + ggml_webnn_shape_js(src1);
        }
        desc += "],\"out\":" + ggml_webnn_shape_js(node) + "}";
    };

    switch (node->op) {
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        {
            const char * fn = node->op == GGML_OP_ADD ? "add" :
                              node->op == GGML_OP_SUB ? "sub" :
                              node->op == GGML_OP_MUL ? "mul" : "div";
            desc_simple(std::string("\"op\":\"") + fn + "\"");
            return true;
        }
        case GGML_OP_MUL_MAT:
            desc_simple("\"op\":\"matmul\"");
            return true;
        case GGML_OP_SCALE:
        {
            float params[2]; // scale, bias
            memcpy(params, node->op_params, sizeof(params));
            desc_simple("\"op\":\"linear\",\"alpha\":" + ggml_webnn_float_js(params[0]) +
                        ",\"beta\":" + ggml_webnn_float_js(params[1]));
            return true;
        }
        case GGML_OP_SOFT_MAX:
        {
            float scale;
            memcpy(&scale, node->op_params, sizeof(scale));
            desc_simple("\"op\":\"softmax\",\"scale\":" + ggml_webnn_float_js(scale));
            return true;
        }
        case GGML_OP_RMS_NORM:
        {
            float eps;
            memcpy(&eps, node->op_params, sizeof(eps));
            desc_simple("\"op\":\"rms_norm\",\"eps\":" + ggml_webnn_float_js(eps));
            return true;
        }
        case GGML_OP_GLU:
        {
            // SWIGLU only (checked by supports_op); silu is applied to the gate
            int32_t swapped;
            memcpy(&swapped, (const int32_t *) node->op_params + 1, sizeof(swapped));
            if (src1) {
                desc_simple("\"op\":\"swiglu_split\",\"gate\":" + std::to_string(swapped ? 1 : 0));
            } else {
                desc_simple("\"op\":\"swiglu\",\"swapped\":" + std::to_string(swapped ? 1 : 0));
            }
            return true;
        }
        case GGML_OP_GET_ROWS:
            // 2D src, 1D i32 indices (enforced by supports_op)
            desc = "{\"op\":\"get_rows\",\"in\":[" + ggml_webnn_shape_js(src0, 2) + "," +
                   ggml_webnn_shape_js(src1, 1) + "],\"dt\":[\"float32\",\"int32\"],\"out\":" +
                   ggml_webnn_shape_js(node, 2) + "}";
            return true;
        case GGML_OP_UNARY:
        {
            if (ggml_get_unary_op(node) == GGML_UNARY_OP_SILU) {
                desc_simple("\"op\":\"silu\"");
                return true;
            }
            const char * fn = ggml_webnn_unary_fn(node);
            if (fn == nullptr) {
                return false;
            }
            desc_simple(std::string("\"op\":\"unary\",\"fn\":\"") + fn + "\"");
            return true;
        }
        case GGML_OP_SQR:
            desc_simple("\"op\":\"sqr\"");
            return true;
        case GGML_OP_SQRT:
            desc_simple("\"op\":\"unary\",\"fn\":\"sqrt\"");
            return true;
        case GGML_OP_LOG:
            desc_simple("\"op\":\"unary\",\"fn\":\"log\"");
            return true;
        case GGML_OP_SIN:
            desc_simple("\"op\":\"unary\",\"fn\":\"sin\"");
            return true;
        case GGML_OP_COS:
            desc_simple("\"op\":\"unary\",\"fn\":\"cos\"");
            return true;
        default:
            return false;
    }
}

//
// backend interface
//

static const char * ggml_backend_webnn_name(ggml_backend_t backend) {
    return GGML_WEBNN_NAME;

    GGML_UNUSED(backend);
}

static void ggml_backend_webnn_free(ggml_backend_t backend) {
    GGML_LOG_INFO("ggml-webnn: %zu ops were dispatched to WebNN\n", g_webnn_n_dispatch);
    delete backend;
}

static enum ggml_status ggml_backend_webnn_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        if (ggml_is_empty(node) || (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        switch (node->op) {
            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                continue;
            case GGML_OP_CPY:
            case GGML_OP_CONT:
            case GGML_OP_DUP:
                // same type + contiguous (checked by supports_op): plain host memcpy
                memcpy(node->data, node->src[0]->data, ggml_nbytes(node->src[0]));
                continue;
            default:
                break;
        }

        std::string desc;
        int n_inputs = 0;
        if (!ggml_webnn_encode_desc(node, desc, n_inputs)) {
            GGML_LOG_ERROR("ggml-webnn: unsupported op in graph: %s\n", ggml_op_desc(node));
            return GGML_STATUS_FAILED;
        }

        WEBNN_LOG_DEBUG("ggml-webnn: dispatch %s\n", desc.c_str());

        const ggml_tensor * src0 = node->src[0];
        const ggml_tensor * src1 = node->src[1];

        const int ret = ggml_webnn_js_dispatch(desc.c_str(),
            src0->data, ggml_nbytes(src0),
            n_inputs > 1 ? src1->data : nullptr,
            n_inputs > 1 ? ggml_nbytes(src1) : 0,
            node->data, ggml_nbytes(node));

        if (ret != 0) {
            GGML_LOG_ERROR("ggml-webnn: dispatch failed for op %s\n", ggml_op_desc(node));
            return GGML_STATUS_FAILED;
        }

        g_webnn_n_dispatch++;
    }

    return GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
}

static struct ggml_backend_i webnn_backend_i = {
    /* .get_name                = */ ggml_backend_webnn_name,
    /* .free                    = */ ggml_backend_webnn_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_webnn_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_webnn_guid(void) {
    static ggml_guid guid = { 0x9c, 0x3e, 0x5a, 0x10, 0x77, 0xb2, 0x4d, 0x8f, 0xa1, 0x06, 0xee, 0x59, 0x21, 0xc4, 0x7b, 0x38 };
    return &guid;
}

ggml_backend_t ggml_backend_webnn_init(void) {
    ggml_backend_reg_t reg = ggml_backend_webnn_reg();
    if (ggml_backend_reg_dev_count(reg) == 0) {
        return nullptr;
    }

    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_webnn_guid(),
        /* .iface   = */ webnn_backend_i,
        /* .device  = */ ggml_backend_reg_dev_get(reg, 0),
        /* .context = */ nullptr,
    };

    return backend;
}

//
// device interface
//

static const char * ggml_backend_webnn_device_get_name(ggml_backend_dev_t dev) {
    return GGML_WEBNN_NAME;

    GGML_UNUSED(dev);
}

static const char * ggml_backend_webnn_device_get_description(ggml_backend_dev_t dev) {
    return "WebNN (navigator.ml)";

    GGML_UNUSED(dev);
}

static void ggml_backend_webnn_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    // no way to query WebNN device memory
    *free  = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_webnn_device_get_type(ggml_backend_dev_t dev) {
    // tensor data stays in host memory and supported ops are dispatched to
    // WebNN, like the BLAS backend
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

static void ggml_backend_webnn_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_webnn_device_get_name(dev);
    props->description = ggml_backend_webnn_device_get_description(dev);
    props->type        = ggml_backend_webnn_device_get_type(dev);
    ggml_backend_webnn_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_webnn_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    return ggml_backend_webnn_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_webnn_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}

static ggml_backend_buffer_t ggml_backend_webnn_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

static bool ggml_webnn_all_f32_contig(const ggml_tensor * op) {
    if (op->type != GGML_TYPE_F32 || !ggml_is_contiguous(op)) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        const ggml_tensor * src = op->src[i];
        if (src == nullptr) {
            break;
        }
        if (src->type != GGML_TYPE_F32 || !ggml_is_contiguous(src)) {
            return false;
        }
    }
    return true;
}

static bool ggml_backend_webnn_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];

    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;

        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        {
            if (!ggml_webnn_all_f32_contig(op)) {
                return false;
            }
            // WebNN broadcasts numpy-style: src1 dims must match or be 1
            // (ggml additionally allows repeat factors, which WebNN cannot do)
            for (int i = 0; i < GGML_MAX_DIMS; i++) {
                if (src1->ne[i] != src0->ne[i] && src1->ne[i] != 1) {
                    return false;
                }
            }
            return true;
        }

        case GGML_OP_MUL_MAT:
        {
            if (!ggml_webnn_all_f32_contig(op)) {
                return false;
            }
            // batch dims must match or broadcast src0 across src1
            for (int i = 2; i < GGML_MAX_DIMS; i++) {
                if (src0->ne[i] != src1->ne[i] && src0->ne[i] != 1) {
                    return false;
                }
            }
            return true;
        }

        case GGML_OP_SCALE:
        case GGML_OP_RMS_NORM:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_LOG:
        case GGML_OP_SIN:
        case GGML_OP_COS:
            return ggml_webnn_all_f32_contig(op);

        case GGML_OP_SOFT_MAX:
        {
            // optional f32 mask; no attention sinks, no alibi
            if (op->src[2] != nullptr) {
                return false;
            }
            float max_bias;
            memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(float));
            if (max_bias != 0.0f || !ggml_webnn_all_f32_contig(op)) {
                return false;
            }
            if (src1 != nullptr) {
                // mask rows must match exactly; higher dims broadcast
                if (src1->ne[0] != src0->ne[0] || src1->ne[1] != src0->ne[1]) {
                    return false;
                }
                for (int i = 2; i < GGML_MAX_DIMS; i++) {
                    if (src1->ne[i] != src0->ne[i] && src1->ne[i] != 1) {
                        return false;
                    }
                }
            }
            return true;
        }

        case GGML_OP_GLU:
            return ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU &&
                   ggml_webnn_all_f32_contig(op) &&
                   (src1 == nullptr || ggml_are_same_shape(src0, src1));

        case GGML_OP_CPY:
        case GGML_OP_CONT:
        case GGML_OP_DUP:
            // same-type contiguous copies are a host memcpy
            return src0->type == op->type &&
                   (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_I32) &&
                   ggml_is_contiguous(src0) && ggml_is_contiguous(op);

        case GGML_OP_GET_ROWS:
            return op->type == GGML_TYPE_F32 &&
                   src0->type == GGML_TYPE_F32 && ggml_is_contiguous(src0) &&
                   src1->type == GGML_TYPE_I32 && ggml_is_contiguous(src1) &&
                   src0->ne[2] == 1 && src0->ne[3] == 1 &&
                   src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1;

        case GGML_OP_UNARY:
            if (!ggml_webnn_all_f32_contig(op)) {
                return false;
            }
            return ggml_get_unary_op(op) == GGML_UNARY_OP_SILU || ggml_webnn_unary_fn(op) != nullptr;

        default:
            return false;
    }

    GGML_UNUSED(dev);
}

static bool ggml_backend_webnn_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft);

    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_webnn_device_i = {
    /* .get_name             = */ ggml_backend_webnn_device_get_name,
    /* .get_description      = */ ggml_backend_webnn_device_get_description,
    /* .get_memory           = */ ggml_backend_webnn_device_get_memory,
    /* .get_type             = */ ggml_backend_webnn_device_get_type,
    /* .get_props            = */ ggml_backend_webnn_device_get_props,
    /* .init_backend         = */ ggml_backend_webnn_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_webnn_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_webnn_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_webnn_device_supports_op,
    /* .supports_buft        = */ ggml_backend_webnn_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

//
// backend reg interface
//

static const char * ggml_backend_webnn_reg_get_name(ggml_backend_reg_t reg) {
    return GGML_WEBNN_NAME;

    GGML_UNUSED(reg);
}

static size_t ggml_backend_webnn_reg_get_device_count(ggml_backend_reg_t reg) {
    return g_webnn_available ? 1 : 0;

    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_webnn_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    GGML_ASSERT(g_webnn_available);

    static ggml_backend_device ggml_backend_webnn_device = {
        /* .iface   = */ ggml_backend_webnn_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    return &ggml_backend_webnn_device;

    GGML_UNUSED(reg);
    GGML_UNUSED(index);
}

static const struct ggml_backend_reg_i ggml_backend_webnn_reg_i = {
    /* .get_name         = */ ggml_backend_webnn_reg_get_name,
    /* .get_device_count = */ ggml_backend_webnn_reg_get_device_count,
    /* .get_device       = */ ggml_backend_webnn_reg_get_device,
    /* .get_proc_address = */ NULL,
};

ggml_backend_reg_t ggml_backend_webnn_reg(void) {
    static bool initialized = false;
    if (!initialized) {
        initialized = true;
        g_webnn_available = ggml_webnn_js_init() != 0;
        if (!g_webnn_available) {
            GGML_LOG_WARN("ggml-webnn: navigator.ml is not available, WebNN backend disabled\n");
        }
    }

    static struct ggml_backend_reg ggml_backend_webnn_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_webnn_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_webnn_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_webnn_reg)

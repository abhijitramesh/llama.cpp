// WebNN backend for ggml
//
// WebNN (navigator.ml) is a graph-based browser API: ops are recorded into an
// MLGraphBuilder, compiled once with build(), and then executed repeatedly with
// dispatch() on MLTensor handles. There is no native C API, so this backend is
// Emscripten-only and talks to the browser through EM_ASYNC_JS bindings (which
// require JSPI or ASYNCIFY to suspend the wasm stack across JS promises).
//
// Design (v2 - whole-graph compilation):
//  - tensor data lives in host (wasm heap) memory, reusing the CPU buffer type
//  - graph_compute translates each scheduler split it receives into a single
//    WebNN MLGraph (compiled once, cached by a pointer-free descriptor that
//    captures ops, dtypes, shapes, params and topology)
//  - graph inputs get pooled MLTensors keyed by host pointer; tensors living
//    in WEIGHTS buffers are uploaded once and reused across dispatches
//  - every node's result is declared a graph output and written back to host
//    memory after dispatch: the scheduler and other backends may read any
//    intermediate tensor across split boundaries
//  - f32 and f16 tensors are supported; mixed matmuls cast up to f32 by
//    default, or down to f16 when globalThis.GGML_WEBNN_FORCE_F16 is set by
//    the embedding page (relevant for NPU/ANE experiments)
//
// Future work: device-resident buffers (skip host writeback), quantized
// weights via dequantizeLinear, ROPE/attention coverage.

#include "ggml-webnn.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <emscripten/emscripten.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#ifdef GGML_WEBNN_DEBUG
#    define WEBNN_LOG_DEBUG(...) GGML_LOG_DEBUG(__VA_ARGS__)
#else
#    define WEBNN_LOG_DEBUG(...) ((void) 0)
#endif

//
// JS bindings
//

// create the MLContext and install the graph/tensor caches on globalThis.
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
            graphs  : new Map(), /* desc string -> {graph, ext, inKeys, outTensors, outFeeds} */
            tpool   : new Map(), /* 'ptr|dt|shape' -> {t: MLTensor, written: bool} */
        };
        /* compile the multi-op graph described by descStr:
           { f16, ext:[{dt,shape}...], nodes:[{op,dt,shape,src,ss,sl?,...params}...], outs:[...] }
           src ref < 0 means external input -(ref+1), otherwise an earlier node index.
           ss[k] is the shape the consumer wants (reshape applied when it differs),
           sl[k] is an optional element offset (1D slice of the producer). */
        S.buildGraph = async function(descStr) {
            const d = JSON.parse(descStr);
            const b = new MLGraphBuilder(S.context);
            const ext = d.ext.map(function(e, i) {
                return b.input('x' + i, { dataType : e.dt, shape : e.shape, dimensions : e.shape });
            });
            const ops = [];
            const nel = function(s) { return s.reduce(function(a, v) { return a * v; }, 1); };
            const same = function(a, c) { return a.length === c.length && a.every(function(v, i) { return v === c[i]; }); };
            const srcDt = function(n, k) {
                const r = n.src[k];
                return r < 0 ? d.ext[-r - 1].dt : d.nodes[r].dt;
            };
            const srcOp = function(n, k, wantDt) {
                const r = n.src[k];
                let op, have;
                if (r < 0) { op = ext[-r - 1]; have = d.ext[-r - 1].shape; }
                else       { op = ops[r];      have = d.nodes[r].shape; }
                const want = n.ss[k];
                const sl = n.sl ? n.sl[k] : null;
                const vp = n.vp ? n.vp[k] : null;
                if (vp !== null && vp !== undefined) {
                    /* strided (permuted-dense) view: flatten, slice the span,
                       restore memory order, then permute to the logical shape */
                    const total = nel(want);
                    if (sl !== null && sl !== undefined) {
                        op = b.reshape(op, [nel(have)]);
                        op = b.slice(op, [sl], [total]);
                    } else if (have.length !== 1 || have[0] !== total) {
                        op = b.reshape(op, [total]);
                    }
                    const vshape = [0, 0, 0, 0];
                    for (let j = 0; j < 4; j++) {
                        vshape[vp[j]] = want[j];
                    }
                    op = b.transpose(b.reshape(op, vshape), { permutation : vp });
                    have = want;
                } else if (sl !== null && sl !== undefined) {
                    op = b.reshape(op, [nel(have)]);
                    op = b.slice(op, [sl], [nel(want)]);
                    have = [nel(want)];
                }
                if (!same(have, want)) {
                    op = b.reshape(op, want);
                }
                if (wantDt && srcDt(n, k) !== wantDt) {
                    op = b.cast(op, wantDt);
                }
                return op;
            };
            for (let i = 0; i < d.nodes.length; i++) {
                const n = d.nodes[i];
                let out;
                switch (n.op) {
                    case 'add': case 'sub': case 'mul': case 'div':
                        out = b[n.op](srcOp(n, 0, n.dt), srcOp(n, 1, n.dt));
                        break;
                    case 'matmul':
                    {
                        /* ggml mul_mat: dst[n,m] = dot(src0 row m, src1 row n)
                           JS shapes are [b3, b2, rows, cols]: matmul(src1, src0^T).
                           GQA grouped broadcast: split src1's batch dim into
                           [shared, group], give src0 a size-1 group axis, and let
                           the rank-5 matmul broadcast (KV heads are read once) */
                        const cdt = n.cdt || 'float32';
                        const w = srcOp(n, 0, cdt);
                        const a = srcOp(n, 1, cdt);
                        const ws = n.ss[0];
                        const as = n.ss[1];
                        if (n.g2 > 1 || n.g3 > 1) {
                            let a5, w5;
                            if (n.g2 > 1) {
                                a5 = b.reshape(a, [as[0], ws[1], n.g2, as[2], as[3]]);
                                w5 = b.reshape(w, [ws[0], ws[1], 1, ws[2], ws[3]]);
                            } else {
                                a5 = b.reshape(a, [ws[0], n.g3, as[1], as[2], as[3]]);
                                w5 = b.reshape(w, [ws[0], 1, ws[1], ws[2], ws[3]]);
                            }
                            out = b.matmul(a5, b.transpose(w5, { permutation : [0, 1, 2, 4, 3] }));
                            out = b.reshape(out, n.shape);
                        } else {
                            out = b.matmul(a, b.transpose(w, { permutation : [0, 1, 3, 2] }));
                        }
                        if (cdt !== n.dt) {
                            out = b.cast(out, n.dt);
                        }
                        break;
                    }
                    case 'unary':
                        out = b[n.fn](srcOp(n, 0, n.dt));
                        break;
                    case 'silu':
                    {
                        const x = srcOp(n, 0, n.dt);
                        out = b.mul(x, b.sigmoid(x));
                        break;
                    }
                    case 'sqr':
                    {
                        const x = srcOp(n, 0, n.dt);
                        out = b.mul(x, x);
                        break;
                    }
                    case 'linear':
                        out = b.linear(srcOp(n, 0, n.dt), { alpha : n.alpha, beta : n.beta });
                        break;
                    case 'softmax':
                    {
                        let t = srcOp(n, 0, n.dt);
                        if (n.scale !== 1) {
                            t = b.linear(t, { alpha : n.scale, beta : 0 });
                        }
                        if (n.src.length > 1) {
                            t = b.add(t, srcOp(n, 1, n.dt)); /* additive attention mask */
                        }
                        out = b.softmax(t, n.shape.length - 1);
                        break;
                    }
                    case 'rms_norm':
                    {
                        const x = srcOp(n, 0, n.dt);
                        const ms = b.reduceMean(b.mul(x, x), { axes : [n.shape.length - 1], keepDimensions : true });
                        const eps = b.constant({ dataType : 'float32', shape : [1], dimensions : [1] },
                                               new Float32Array([n.eps]));
                        out = b.div(x, b.sqrt(b.add(ms, eps)));
                        break;
                    }
                    case 'swiglu_split':
                    {
                        const g = srcOp(n, n.gate, n.dt);
                        const u = srcOp(n, 1 - n.gate, n.dt);
                        out = b.mul(b.mul(g, b.sigmoid(g)), u);
                        break;
                    }
                    case 'swiglu':
                    {
                        const x = srcOp(n, 0, n.dt);
                        const s = n.ss[0];
                        const half = s[3] / 2;
                        const sizes = [s[0], s[1], s[2], half];
                        const g = b.slice(x, [0, 0, 0, n.swapped ? half : 0], sizes);
                        const u = b.slice(x, [0, 0, 0, n.swapped ? 0 : half], sizes);
                        out = b.mul(b.mul(g, b.sigmoid(g)), u);
                        break;
                    }
                    case 'rope':
                    {
                        /* theta[t,i] = pos[t] * freqs[i]; rotate pairs by (cos,sin)*attn_factor.
                           mode 0 (NORM) rotates interleaved pairs, mode 2 (NEOX) rotates halves */
                        const x = srcOp(n, 0, n.dt);          /* [ns, t, h, d] */
                        const s = n.shape;
                        const half = s[3] / 2;
                        const pos = b.cast(b.reshape(srcOp(n, 1, null), [1, s[1], 1, 1]), 'float32');
                        const freqs = b.constant({ dataType : 'float32', shape : [1, 1, 1, half], dimensions : [1, 1, 1, half] },
                                                 new Float32Array(n.freqs));
                        const theta = b.mul(pos, freqs);      /* [1, t, 1, half] */
                        const cosv = b.linear(b.cos(theta), { alpha : n.af, beta : 0 });
                        const sinv = b.linear(b.sin(theta), { alpha : n.af, beta : 0 });
                        const hsize = [s[0], s[1], s[2], half];
                        let xe, xo;
                        if (n.mode === 2) {
                            xe = b.slice(x, [0, 0, 0, 0], hsize);
                            xo = b.slice(x, [0, 0, 0, half], hsize);
                        } else {
                            const x5 = b.reshape(x, [s[0], s[1], s[2], half, 2]);
                            xe = b.reshape(b.slice(x5, [0, 0, 0, 0, 0], [s[0], s[1], s[2], half, 1]), hsize);
                            xo = b.reshape(b.slice(x5, [0, 0, 0, 0, 1], [s[0], s[1], s[2], half, 1]), hsize);
                        }
                        const oe = b.sub(b.mul(xe, cosv), b.mul(xo, sinv));
                        const oo = b.add(b.mul(xe, sinv), b.mul(xo, cosv));
                        if (n.mode === 2) {
                            out = b.concat([oe, oo], 3);
                        } else {
                            const h5 = [s[0], s[1], s[2], half, 1];
                            out = b.reshape(b.concat([b.reshape(oe, h5), b.reshape(oo, h5)], 4), s);
                        }
                        break;
                    }
                    case 'flash_attn':
                    {
                        /* softmax(q @ k^T * scale + mask) @ v, heads as batch dim.
                           q [ns,nh,nt,d], k/v [ns,nhkv,kv,d|dv], mask [ns,1,nt,kv].
                           GQA (g>1): q gains a [nhkv, g] split, k/v a size-1 group
                           axis, and rank-5 broadcasting shares each KV head */
                        const cdt = n.cdt || 'float32';
                        let q = srcOp(n, 0, cdt);
                        let kk = srcOp(n, 1, cdt);
                        let v = srcOp(n, 2, cdt);
                        const g = n.g || 1;
                        const qs = n.ss[0];
                        const ks = n.ss[1];
                        let kq;
                        if (g > 1) {
                            q = b.reshape(q, [qs[0], ks[1], g, qs[2], qs[3]]);
                            kk = b.reshape(kk, [ks[0], ks[1], 1, ks[2], ks[3]]);
                            kq = b.matmul(q, b.transpose(kk, { permutation : [0, 1, 2, 4, 3] }));
                        } else {
                            kq = b.matmul(q, b.transpose(kk, { permutation : [0, 1, 3, 2] }));
                        }
                        if (n.scale !== 1) {
                            kq = b.linear(kq, { alpha : n.scale, beta : 0 });
                        }
                        if (n.src.length > 3) {
                            let m = srcOp(n, 3, cdt);
                            if (g > 1) {
                                const ms = n.ss[3];
                                m = b.reshape(m, [ms[0], 1, 1, ms[2], ms[3]]);
                            }
                            kq = b.add(kq, m);
                        }
                        const probs = b.softmax(kq, g > 1 ? 4 : 3);
                        let o;
                        if (g > 1) {
                            const vs = n.ss[2];
                            v = b.reshape(v, [vs[0], vs[1], 1, vs[2], vs[3]]);
                            o = b.matmul(probs, v);                            /* [ns, nhkv, g, nt, dv] */
                            o = b.reshape(o, [qs[0], qs[1], qs[2], vs[3]]);    /* [ns, nh, nt, dv] */
                        } else {
                            o = b.matmul(probs, v);
                        }
                        o = b.transpose(o, { permutation : [0, 2, 1, 3] }); /* dst is [ns, nt, nh, dv] */
                        out = cdt !== n.dt ? b.cast(o, n.dt) : o;
                        break;
                    }
                    case 'get_rows':
                    {
                        out = b.gather(srcOp(n, 0, null), srcOp(n, 1, null), { axis : 0 });
                        if (srcDt(n, 0) !== n.dt) {
                            out = b.cast(out, n.dt);
                        }
                        break;
                    }
                    case 'copy':
                    {
                        const x = srcOp(n, 0, null);
                        out = srcDt(n, 0) !== n.dt ? b.cast(x, n.dt) : b.identity(x);
                        if (!same(n.ss[0], n.shape)) {
                            out = b.reshape(out, n.shape); /* CPY may change the logical shape */
                        }
                        break;
                    }
                    default:
                        throw new Error('unknown op ' + n.op);
                }
                ops.push(out);
            }
            const outputsObj = {};
            for (const i of d.outs) {
                outputsObj['n' + i] = ops[i];
            }
            const graph = await b.build(outputsObj);
            const outTensors = [];
            const outFeeds = {};
            for (const i of d.outs) {
                const n = d.nodes[i];
                const t = await S.context.createTensor(
                    { dataType : n.dt, shape : n.shape, dimensions : n.shape, readable : true });
                outTensors.push(t);
                outFeeds['n' + i] = t;
            }
            const entry = {
                graph : graph, ext : d.ext, outTensors : outTensors, outFeeds : outFeeds,
                inKeys : d.ext.map(function(e) { return e.dt + '|' + e.shape.join('x'); }),
            };
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

// whether the embedding page requests f16 matmul compute (ANE-friendly)
EM_JS(int, ggml_webnn_js_force_f16, (), {
    return globalThis.GGML_WEBNN_FORCE_F16 ? 1 : 0;
});

// execute one compiled graph: upload changed inputs, dispatch, read all
// outputs back into the wasm heap. in_tab is n_in x [ptr, nbytes, flags]
// (flags bit0 = weight), out_tab is n_out x [ptr, nbytes]. returns 0 on success.
EM_ASYNC_JS(int, ggml_webnn_js_graph_dispatch,
            (const char * desc, void * in_tab, int n_in, void * out_tab, int n_out), {
    const S = globalThis.__ggml_webnn;
    try {
        /* under JSPI the wasm ABI passes these as BigInt - coerce before use */
        const descStr = UTF8ToString(Number(desc));
        const inT = Number(in_tab) >> 2;
        const outT = Number(out_tab) >> 2;
        n_in = Number(n_in);
        n_out = Number(n_out);

        let entry = S.graphs.get(descStr);
        if (!entry) {
            entry = await S.buildGraph(descStr);
        }

        const feeds = {};
        for (let i = 0; i < n_in; i++) {
            const ptr = HEAPU32[inT + i*3];
            const nb = HEAPU32[inT + i*3 + 1];
            const fl = HEAPU32[inT + i*3 + 2];
            const key = ptr + '|' + entry.inKeys[i];
            let rec = S.tpool.get(key);
            if (!rec) {
                const e = entry.ext[i];
                rec = { t : await S.context.createTensor(
                            { dataType : e.dt, shape : e.shape, dimensions : e.shape, writable : true }),
                        written : false };
                S.tpool.set(key, rec);
            }
            const isWeight = (fl & 1) !== 0;
            if (!isWeight || !rec.written) {
                /* writeTensor copies synchronously, so a heap view is safe here */
                S.context.writeTensor(rec.t, HEAPU8.subarray(ptr, ptr + nb));
                rec.written = true;
            }
            feeds['x' + i] = rec.t;
        }

        S.context.dispatch(entry.graph, feeds, entry.outFeeds);

        const bufs = await Promise.all(entry.outTensors.map(function(t) { return S.context.readTensor(t); }));
        for (let j = 0; j < n_out; j++) {
            /* HEAPU8/U32 are re-read after the awaits - memory may have grown */
            HEAPU8.set(new Uint8Array(bufs[j]), HEAPU32[outT + j*2]);
        }
        return 0;
    } catch (e) {
        console.error('ggml-webnn: graph dispatch failed:', e, UTF8ToString(Number(desc)).slice(0, 4000));
        return 1;
    }
});

static bool   g_webnn_available  = false;
static bool   g_webnn_force_f16  = false;
static size_t g_webnn_n_ops      = 0; // ggml nodes executed via WebNN
static size_t g_webnn_n_dispatch = 0; // compiled-graph dispatches
static std::map<std::string, size_t> g_webnn_op_tally; // per-op execution counts

//
// graph descriptor encoding
//

static const char * ggml_webnn_dt(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32: return "float32";
        case GGML_TYPE_F16: return "float16";
        case GGML_TYPE_I32: return "int32";
        default:            return nullptr;
    }
}

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

// check whether t is a permutation of a dense block of prod(ne) elements
// starting at t->data (e.g. ggml_permute views, KV cache head-interleaved
// views). fills the JS transpose permutation that restores logical order.
static bool ggml_webnn_permuted_contig(const ggml_tensor * t, int perm_out[GGML_MAX_DIMS]) {
    if (ggml_type_size(t->type) == 0 || ggml_blck_size(t->type) != 1) {
        return false;
    }
    // dims with ne>1, ordered by stride ascending, must form a dense chain
    int dims[GGML_MAX_DIMS];
    int n = 0;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (t->ne[i] > 1) {
            dims[n++] = i;
        }
    }
    for (int a = 0; a < n; a++) {
        for (int c = a + 1; c < n; c++) {
            if (t->nb[dims[c]] < t->nb[dims[a]]) {
                int tmp = dims[a]; dims[a] = dims[c]; dims[c] = tmp;
            }
        }
    }
    size_t expect = ggml_type_size(t->type);
    for (int a = 0; a < n; a++) {
        if (t->nb[dims[a]] != expect) {
            return false;
        }
        expect *= t->ne[dims[a]];
    }
    // memory order, outermost first: ne==1 dims placed outermost (stride is moot)
    int o[GGML_MAX_DIMS];
    int idx = 0;
    for (int i = GGML_MAX_DIMS - 1; i >= 0; i--) {
        if (t->ne[i] == 1) {
            o[idx++] = i;
        }
    }
    for (int a = n - 1; a >= 0; a--) {
        o[idx++] = dims[a];
    }
    // JS transpose: output (logical) axis j is ggml dim 3-j; find it in memory order
    for (int j = 0; j < GGML_MAX_DIMS; j++) {
        for (int k = 0; k < GGML_MAX_DIMS; k++) {
            if (o[k] == GGML_MAX_DIMS - 1 - j) {
                perm_out[j] = k;
                break;
            }
        }
    }
    return true;
}

// contiguous, or a strided view we can translate (reshape+transpose)
static bool ggml_webnn_view_ok(const ggml_tensor * t) {
    if (ggml_is_contiguous(t)) {
        return true;
    }
    int perm[GGML_MAX_DIMS];
    return ggml_webnn_permuted_contig(t, perm);
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

// op-specific JSON fields ("op":... plus parameters), false for unhandled ops
static bool ggml_webnn_op_fields(const ggml_tensor * node, std::string & fields) {
    switch (node->op) {
        case GGML_OP_ADD: fields = "\"op\":\"add\""; return true;
        case GGML_OP_SUB: fields = "\"op\":\"sub\""; return true;
        case GGML_OP_MUL: fields = "\"op\":\"mul\""; return true;
        case GGML_OP_DIV: fields = "\"op\":\"div\""; return true;
        case GGML_OP_MUL_MAT:
        {
            fields = std::string("\"op\":\"matmul\",\"cdt\":\"") +
                     (g_webnn_force_f16 ? "float16" : "float32") + "\"";
            // grouped batch broadcast (GQA): consecutive src1 batches share a src0 batch
            const ggml_tensor * s0 = node->src[0];
            const ggml_tensor * s1 = node->src[1];
            if (s0->ne[2] > 1 && s1->ne[2] != s0->ne[2]) {
                fields += ",\"g2\":" + std::to_string(s1->ne[2] / s0->ne[2]);
            }
            if (s0->ne[3] > 1 && s1->ne[3] != s0->ne[3]) {
                fields += ",\"g3\":" + std::to_string(s1->ne[3] / s0->ne[3]);
            }
            return true;
        }
        case GGML_OP_SCALE:
        {
            float params[2]; // scale, bias
            memcpy(params, node->op_params, sizeof(params));
            fields = "\"op\":\"linear\",\"alpha\":" + ggml_webnn_float_js(params[0]) +
                     ",\"beta\":" + ggml_webnn_float_js(params[1]);
            return true;
        }
        case GGML_OP_SOFT_MAX:
        {
            float scale;
            memcpy(&scale, node->op_params, sizeof(scale));
            fields = "\"op\":\"softmax\",\"scale\":" + ggml_webnn_float_js(scale);
            return true;
        }
        case GGML_OP_RMS_NORM:
        {
            float eps;
            memcpy(&eps, node->op_params, sizeof(eps));
            fields = "\"op\":\"rms_norm\",\"eps\":" + ggml_webnn_float_js(eps);
            return true;
        }
        case GGML_OP_GLU:
        {
            // SWIGLU only (checked by supports_op); silu is applied to the gate
            int32_t swapped;
            memcpy(&swapped, (const int32_t *) node->op_params + 1, sizeof(swapped));
            if (node->src[1]) {
                fields = "\"op\":\"swiglu_split\",\"gate\":" + std::to_string(swapped ? 1 : 0);
            } else {
                fields = "\"op\":\"swiglu\",\"swapped\":" + std::to_string(swapped ? 1 : 0);
            }
            return true;
        }
        case GGML_OP_ROPE:
        {
            // mode/freq params checked by supports_op; ext_factor==0, no freq_factors
            const int32_t * ip = (const int32_t *) node->op_params;
            const int n_dims = ip[1];
            const int mode   = ip[2];
            float freq_base, freq_scale, attn_factor;
            memcpy(&freq_base,   (const float *) node->op_params + 5, sizeof(float));
            memcpy(&freq_scale,  (const float *) node->op_params + 6, sizeof(float));
            memcpy(&attn_factor, (const float *) node->op_params + 8, sizeof(float));
            fields = "\"op\":\"rope\",\"mode\":" + std::to_string(mode) +
                     ",\"af\":" + ggml_webnn_float_js(attn_factor) + ",\"freqs\":[";
            for (int i = 0; i < n_dims/2; i++) {
                if (i > 0) {
                    fields += ",";
                }
                fields += ggml_webnn_float_js(freq_scale * powf(freq_base, -2.0f*i/n_dims));
            }
            fields += "]";
            return true;
        }
        case GGML_OP_FLASH_ATTN_EXT:
        {
            float scale;
            memcpy(&scale, node->op_params, sizeof(scale));
            fields = "\"op\":\"flash_attn\",\"scale\":" + ggml_webnn_float_js(scale) +
                     ",\"cdt\":\"" + (g_webnn_force_f16 ? "float16" : "float32") + "\"";
            if (node->src[0]->ne[2] != node->src[1]->ne[2]) {
                // GQA: query heads per KV head
                fields += ",\"g\":" + std::to_string(node->src[0]->ne[2] / node->src[1]->ne[2]);
            }
            return true;
        }
        case GGML_OP_GET_ROWS: fields = "\"op\":\"get_rows\""; return true;
        case GGML_OP_CPY:
        case GGML_OP_CONT:
        case GGML_OP_DUP:      fields = "\"op\":\"copy\""; return true;
        case GGML_OP_UNARY:
        {
            if (ggml_get_unary_op(node) == GGML_UNARY_OP_SILU) {
                fields = "\"op\":\"silu\"";
                return true;
            }
            const char * fn = ggml_webnn_unary_fn(node);
            if (fn == nullptr) {
                return false;
            }
            fields = std::string("\"op\":\"unary\",\"fn\":\"") + fn + "\"";
            return true;
        }
        case GGML_OP_SQR:  fields = "\"op\":\"sqr\"";                       return true;
        case GGML_OP_SQRT: fields = "\"op\":\"unary\",\"fn\":\"sqrt\"";     return true;
        case GGML_OP_LOG:  fields = "\"op\":\"unary\",\"fn\":\"log\"";      return true;
        case GGML_OP_SIN:  fields = "\"op\":\"unary\",\"fn\":\"sin\"";      return true;
        case GGML_OP_COS:  fields = "\"op\":\"unary\",\"fn\":\"cos\"";      return true;
        default:
            return false;
    }
}

static bool ggml_webnn_is_noop(const ggml_tensor * node) {
    switch (node->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
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
    GGML_LOG_INFO("ggml-webnn: %zu ops were executed in %zu WebNN graph dispatches\n",
                  g_webnn_n_ops, g_webnn_n_dispatch);
    std::string tally;
    for (const auto & kv : g_webnn_op_tally) {
        tally += " " + kv.first + ":" + std::to_string(kv.second);
    }
    GGML_LOG_INFO("ggml-webnn: op tally:%s\n", tally.c_str());
    delete backend;
}

static enum ggml_status ggml_backend_webnn_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    // 1. collect the compute nodes of this split
    std::vector<ggml_tensor *> nodes;
    std::map<const ggml_tensor *, int> node_idx;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if (ggml_webnn_is_noop(node) || ggml_is_empty(node) || (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        node_idx[node] = (int) nodes.size();
        nodes.push_back(node);
    }
    if (nodes.empty()) {
        return GGML_STATUS_SUCCESS;
    }

    // 2. translate into a graph descriptor (doubles as the compile cache key)
    std::vector<std::string> ext_descs;
    std::vector<uint32_t>    in_tab; // [ptr, nbytes, flags] per external input
    std::map<std::string, int> ext_idx;

    // resolve a src tensor to an operand reference:
    //   ref < 0  -> external input -(ref+1)  (data read from host at dispatch)
    //   ref >= 0 -> node index in this split (+ optional 1D element slice)
    // non-contiguous (permuted-dense) views additionally get a JS transpose
    // permutation in vp_json; their external inputs are declared flat
    auto resolve = [&](const ggml_tensor * t, int rank, int & ref, int64_t & slice_off, std::string & vp_json) -> bool {
        // walk the view chain, stopping at the FIRST tensor that is a compute
        // node of this split: in-place ops are views of their src0, so walking
        // straight to the root would skip past the producing node
        const ggml_tensor * owner = t;
        auto it = node_idx.find(owner);
        while (it == node_idx.end() && owner->view_src) {
            owner = owner->view_src;
            it = node_idx.find(owner);
        }
        slice_off = -1;
        vp_json = "null";

        const bool contig = ggml_is_contiguous(t);
        if (!contig) {
            int perm[GGML_MAX_DIMS];
            if (rank != 4 || !ggml_webnn_permuted_contig(t, perm)) {
                return false;
            }
            vp_json = "[" + std::to_string(perm[0]) + "," + std::to_string(perm[1]) + "," +
                      std::to_string(perm[2]) + "," + std::to_string(perm[3]) + "]";
        }

        if (it != node_idx.end()) {
            if (owner->type != t->type) {
                return false; // type-punning views are not supported
            }
            ref = it->second;
            const size_t off_bytes = (const char *) t->data - (const char *) owner->data;
            if (off_bytes != 0 || ggml_nelements(t) != ggml_nelements(owner)) {
                if (off_bytes % ggml_type_size(t->type) != 0) {
                    return false;
                }
                slice_off = (int64_t) (off_bytes / ggml_type_size(t->type));
            }
            return true;
        }
        // external input (leaf, weight, or a tensor computed outside this split);
        // strided views are declared as their flat dense span
        const char * dt = ggml_webnn_dt(t->type);
        if (dt == nullptr) {
            return false;
        }
        const std::string shape = contig ? ggml_webnn_shape_js(t, rank)
                                         : "[" + std::to_string(ggml_nelements(t)) + "]";
        const size_t nbytes = contig ? ggml_nbytes(t)
                                     : (size_t) ggml_nelements(t) * ggml_type_size(t->type);
        char key[64];
        snprintf(key, sizeof(key), "%p|%s|", t->data, dt);
        const std::string k = key + shape;
        auto eit = ext_idx.find(k);
        int idx;
        if (eit != ext_idx.end()) {
            idx = eit->second;
        } else {
            idx = (int) ext_descs.size();
            ext_idx[k] = idx;
            ext_descs.push_back("{\"dt\":\"" + std::string(dt) + "\",\"shape\":" + shape + "}");
            uint32_t flags = 0;
            if (t->buffer && ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                flags |= 1;
            }
            in_tab.push_back((uint32_t) (uintptr_t) t->data);
            in_tab.push_back((uint32_t) nbytes);
            in_tab.push_back(flags);
        }
        ref = -(idx + 1);
        return true;
    };

    std::string desc = "{\"f16\":";
    desc += g_webnn_force_f16 ? "1" : "0";
    desc += ",\"nodes\":[";

    for (size_t i = 0; i < nodes.size(); i++) {
        const ggml_tensor * node = nodes[i];

        std::string fields;
        if (!ggml_webnn_op_fields(node, fields)) {
            GGML_LOG_ERROR("ggml-webnn: unsupported op in graph: %s\n", ggml_op_desc(node));
            return GGML_STATUS_FAILED;
        }

        const int node_rank = node->op == GGML_OP_GET_ROWS ? 2 : 4;

        // CPY carries its destination alias as src[1] - it is not a data input
        const int n_srcs = node->op == GGML_OP_CPY ? 1 : GGML_MAX_SRC;

        std::string src_json = "[";
        std::string ss_json  = "[";
        std::string sl_json  = "[";
        std::string vp_json  = "[";
        for (int k = 0; k < n_srcs && node->src[k]; k++) {
            // per-src rank: GET_ROWS uses rank-2 data + rank-1 indices,
            // ROPE positions are a rank-1 i32 vector
            int rank = 4;
            if (node->op == GGML_OP_GET_ROWS) {
                rank = k == 0 ? 2 : 1;
            } else if (node->op == GGML_OP_ROPE && k == 1) {
                rank = 1;
            }
            int ref;
            int64_t slice_off;
            std::string vp;
            if (!resolve(node->src[k], rank, ref, slice_off, vp)) {
                GGML_LOG_ERROR("ggml-webnn: cannot resolve src %d of %s\n", k, ggml_op_desc(node));
                return GGML_STATUS_FAILED;
            }
            if (k > 0) {
                src_json += ",";
                ss_json += ",";
                sl_json += ",";
                vp_json += ",";
            }
            src_json += std::to_string(ref);
            ss_json  += ggml_webnn_shape_js(node->src[k], rank);
            sl_json  += slice_off < 0 ? "null" : std::to_string(slice_off);
            vp_json  += vp;
        }
        src_json += "]";
        ss_json += "]";
        sl_json += "]";
        vp_json += "]";

        if (i > 0) {
            desc += ",";
        }
        desc += "{" + fields +
                ",\"dt\":\"" + ggml_webnn_dt(node->type) + "\"" +
                ",\"shape\":" + ggml_webnn_shape_js(node, node_rank) +
                ",\"src\":" + src_json +
                ",\"ss\":" + ss_json +
                ",\"sl\":" + sl_json +
                ",\"vp\":" + vp_json + "}";
    }

    // 3. every node is materialized back to host memory
    desc += "],\"outs\":[";
    std::vector<uint32_t> out_tab;
    for (size_t i = 0; i < nodes.size(); i++) {
        if (i > 0) {
            desc += ",";
        }
        desc += std::to_string(i);
        out_tab.push_back((uint32_t) (uintptr_t) nodes[i]->data);
        out_tab.push_back((uint32_t) ggml_nbytes(nodes[i]));
    }
    desc += "],\"ext\":[";
    for (size_t i = 0; i < ext_descs.size(); i++) {
        if (i > 0) {
            desc += ",";
        }
        desc += ext_descs[i];
    }
    desc += "]}";

    WEBNN_LOG_DEBUG("ggml-webnn: dispatch %d nodes, %d inputs\n", (int) nodes.size(), (int) ext_descs.size());

    const int ret = ggml_webnn_js_graph_dispatch(desc.c_str(),
        in_tab.data(), (int) ext_descs.size(),
        out_tab.data(), (int) nodes.size());

    if (ret != 0) {
        GGML_LOG_ERROR("ggml-webnn: graph dispatch failed (%d nodes)\n", (int) nodes.size());
        return GGML_STATUS_FAILED;
    }

    g_webnn_n_ops += nodes.size();
    g_webnn_n_dispatch++;
    for (const ggml_tensor * node : nodes) {
        g_webnn_op_tally[ggml_op_desc(node)]++;
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

static bool ggml_webnn_is_float(ggml_type t) {
    return t == GGML_TYPE_F32 || t == GGML_TYPE_F16;
}

// op + all srcs: float dtype (f32/f16) and contiguous
static bool ggml_webnn_all_float_contig(const ggml_tensor * op) {
    if (!ggml_webnn_is_float(op->type) || !ggml_is_contiguous(op)) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        const ggml_tensor * src = op->src[i];
        if (src == nullptr) {
            break;
        }
        if (!ggml_webnn_is_float(src->type) || !ggml_is_contiguous(src)) {
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
            if (!ggml_webnn_all_float_contig(op) || src0->type != op->type || src1->type != op->type) {
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
            // f32/f16 srcs in any combination (strided views allowed), f32 dst
            if (op->type != GGML_TYPE_F32 || !ggml_is_contiguous(op)) {
                return false;
            }
            if (!ggml_webnn_is_float(src0->type) || !ggml_webnn_view_ok(src0) ||
                !ggml_webnn_is_float(src1->type) || !ggml_webnn_view_ok(src1)) {
                return false;
            }
            // batch dims: equal, plain broadcast (src0 dim 1), or grouped
            // broadcast (GQA, src1 a multiple of src0) in at most one dim
            int n_grouped = 0;
            for (int i = 2; i < GGML_MAX_DIMS; i++) {
                if (src1->ne[i] == src0->ne[i] || src0->ne[i] == 1) {
                    continue;
                }
                if (src1->ne[i] % src0->ne[i] == 0) {
                    n_grouped++;
                    continue;
                }
                return false;
            }
            return n_grouped <= 1;
        }

        case GGML_OP_ROPE:
        {
            // f32, full-width rotation, NORM/NEOX, no yarn, no freq_factors
            if (op->src[2] != nullptr || op->type != GGML_TYPE_F32 || src0->type != GGML_TYPE_F32) {
                return false;
            }
            const int32_t * ip = (const int32_t *) op->op_params;
            const int n_dims = ip[1];
            const int mode   = ip[2];
            float ext_factor;
            memcpy(&ext_factor, (const float *) op->op_params + 7, sizeof(float));
            return (mode == GGML_ROPE_TYPE_NORMAL || mode == GGML_ROPE_TYPE_NEOX) &&
                   ext_factor == 0.0f &&
                   n_dims == src0->ne[0] &&
                   ggml_webnn_view_ok(src0) &&
                   src1->type == GGML_TYPE_I32 && ggml_is_contiguous(src1) &&
                   ggml_is_contiguous(op);
        }

        case GGML_OP_FLASH_ATTN_EXT:
        {
            const ggml_tensor * v    = op->src[2];
            const ggml_tensor * mask = op->src[3];
            if (op->src[4] != nullptr) {
                return false; // attention sinks
            }
            float max_bias, logit_softcap;
            memcpy(&max_bias,      (const float *) op->op_params + 1, sizeof(float));
            memcpy(&logit_softcap, (const float *) op->op_params + 2, sizeof(float));
            if (max_bias != 0.0f || logit_softcap != 0.0f) {
                return false;
            }
            if (op->type != GGML_TYPE_F32 || !ggml_is_contiguous(op)) {
                return false;
            }
            if (src0->type != GGML_TYPE_F32 || !ggml_webnn_view_ok(src0)) {
                return false;
            }
            if (!ggml_webnn_is_float(src1->type) || v->type != src1->type ||
                !ggml_webnn_view_ok(src1) || !ggml_webnn_view_ok(v)) {
                return false;
            }
            // query heads must be a multiple of KV heads (GQA via rank-5 broadcast)
            if (src0->ne[2] % src1->ne[2] != 0 || src1->ne[2] != v->ne[2]) {
                return false;
            }
            if (mask != nullptr) {
                if (mask->type != GGML_TYPE_F16 || !ggml_is_contiguous(mask)) {
                    return false;
                }
                if (mask->ne[0] != src1->ne[1] || mask->ne[1] != src0->ne[1] || mask->ne[2] != 1 ||
                    (mask->ne[3] != src0->ne[3] && mask->ne[3] != 1)) {
                    return false;
                }
            }
            return true;
        }

        case GGML_OP_SCALE:
        case GGML_OP_RMS_NORM:
            return op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(src0) && ggml_is_contiguous(op);

        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_LOG:
        case GGML_OP_SIN:
        case GGML_OP_COS:
            return src0->type == op->type && ggml_webnn_all_float_contig(op);

        case GGML_OP_SOFT_MAX:
        {
            // optional f32 mask; no attention sinks, no alibi; f32 only
            if (op->src[2] != nullptr || op->type != GGML_TYPE_F32 || src0->type != GGML_TYPE_F32) {
                return false;
            }
            float max_bias;
            memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(float));
            if (max_bias != 0.0f || !ggml_is_contiguous(src0) || !ggml_is_contiguous(op)) {
                return false;
            }
            if (src1 != nullptr) {
                if (src1->type != GGML_TYPE_F32 || !ggml_is_contiguous(src1)) {
                    return false;
                }
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
                   ggml_webnn_all_float_contig(op) &&
                   src0->type == op->type &&
                   (src1 == nullptr || (src1->type == op->type && ggml_are_same_shape(src0, src1)));

        case GGML_OP_CPY:
        case GGML_OP_CONT:
        case GGML_OP_DUP:
            // float<->float copies (with cast), or same-type i32
            if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(op)) {
                return false;
            }
            if (ggml_webnn_is_float(src0->type) && ggml_webnn_is_float(op->type)) {
                return true;
            }
            return src0->type == GGML_TYPE_I32 && op->type == GGML_TYPE_I32;

        case GGML_OP_GET_ROWS:
            return op->type == GGML_TYPE_F32 &&
                   ggml_webnn_is_float(src0->type) && ggml_is_contiguous(src0) &&
                   src1->type == GGML_TYPE_I32 && ggml_is_contiguous(src1) &&
                   src0->ne[2] == 1 && src0->ne[3] == 1 &&
                   src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1;

        case GGML_OP_UNARY:
            if (src0->type != op->type || !ggml_webnn_all_float_contig(op)) {
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
        if (g_webnn_available) {
            g_webnn_force_f16 = ggml_webnn_js_force_f16() != 0;
            if (g_webnn_force_f16) {
                GGML_LOG_INFO("ggml-webnn: matmuls will be computed in f16 (GGML_WEBNN_FORCE_F16)\n");
            }
        } else {
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

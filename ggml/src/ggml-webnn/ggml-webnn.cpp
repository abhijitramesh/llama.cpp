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

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
            graphs  : new Map(), /* desc string -> {graph, ext, inKeys, outInfo} */
            tpool   : new Map(), /* 'ptr|dt|shape' -> {t: MLTensor, written: bool} */
            resident : new Map(), /* 'r'+rootPtr -> {cur, alt, dt, shape} - device-resident regions (KV caches) */
            chained : new Map(), /* 'c'+ptr -> {t, epoch, dt, shape, p} - segment-boundary tensors kept on device */
            chainEpoch : 0,      /* chained records are only trusted within one graph_compute call */
            pending : [],        /* enqueued readbacks {promise, ptr}, flushed in enqueue order */
            pendingPtrs : new Set(),
            canScatter : !globalThis.GGML_WEBNN_NO_SCATTER &&
                         typeof MLGraphBuilder !== 'undefined' && typeof MLGraphBuilder.prototype.scatterND === 'function',
        };
        /* await all pending readbacks, write to the heap in enqueue order
           (in-place chains alias one pointer - the last value must win) */
        S.flush = async function() {
            const pend = S.pending;
            S.pending = [];
            S.pendingPtrs = new Set();
            const bufs = await Promise.all(pend.map(function(p) { return p.promise; }));
            for (let i = 0; i < bufs.length; i++) {
                HEAPU8.set(new Uint8Array(bufs[i]), pend[i].ptr);
            }
        };
        S.f16 = function(h) {
            const s = (h & 0x8000) ? -1 : 1;
            const e = (h >> 10) & 0x1F;
            const m = h & 0x3FF;
            if (e === 0) {
                return s * m * Math.pow(2, -24);
            }
            if (e === 31) {
                return m ? NaN : s * Infinity;
            }
            return s * (1024 + m) * Math.pow(2, e - 25);
        };
        S.qcache = new Map(); /* 'ptr|fmt|shape' -> repacked weight/scale arrays */
        /* repack a ggml-quantized weight [n, k] into dequantizeLinear constants.
           q4_0 blocks: f16 scale + 16 bytes (low nibbles = elems 0..15, high =
           16..31), value = (q - 8) * scale. q8_0: f16 scale + 32 int8. */
        S.dequantConst = function(b, q, shape, ptr, nbytes) {
            const n = shape[0], k = shape[1], nb = k / 32;
            const key = ptr + '|' + q + '|' + n + 'x' + k;
            let rec = S.qcache.get(key);
            if (q === 'q4_k') {
                /* super-blocks of 256: 2 f16 (d, dmin), 12 bytes of 6-bit
                   scales/mins for 8 sub-blocks, 128 bytes of 4-bit q.
                   value = (d*sc)*q - (dmin*m) per 32-value sub-block */
                if (!rec) {
                    const raw = HEAPU8.subarray(ptr, ptr + nbytes);
                    const nsb = k / 256;
                    const packed = new Uint8Array(n * k / 2);
                    const scales = new Float32Array(n * nb);
                    const mins = new Float32Array(n * nb);
                    const vals = new Uint8Array(256);
                    for (let r = 0; r < n; r++) {
                        for (let sb = 0; sb < nsb; sb++) {
                            const off = (r * nsb + sb) * 144;
                            const d = S.f16(raw[off] | (raw[off + 1] << 8));
                            const dmin = S.f16(raw[off + 2] | (raw[off + 3] << 8));
                            const sc = off + 4; /* 12 bytes of packed 6-bit scales/mins */
                            const subBase = r * nb + sb * 8;
                            for (let j = 0; j < 8; j++) {
                                let scv, mv;
                                if (j < 4) {
                                    scv = raw[sc + j] & 63;
                                    mv = raw[sc + j + 4] & 63;
                                } else {
                                    scv = (raw[sc + j + 4] & 0xF) | ((raw[sc + j - 4] >> 6) << 4);
                                    mv = (raw[sc + j + 4] >> 4) | ((raw[sc + j] >> 6) << 4);
                                }
                                scales[subBase + j] = d * scv;
                                mins[subBase + j] = dmin * mv;
                            }
                            /* per 64 values: 32 q-bytes; low nibbles -> first 32,
                               high nibbles -> next 32 */
                            const qs = off + 16;
                            for (let pair = 0; pair < 4; pair++) {
                                const qb = qs + 32 * pair;
                                const vb = 64 * pair;
                                for (let l = 0; l < 32; l++) {
                                    vals[vb + l] = raw[qb + l] & 0xF;
                                    vals[vb + 32 + l] = raw[qb + l] >> 4;
                                }
                            }
                            const base = (r * k + sb * 256) >> 1;
                            for (let m2 = 0; m2 < 128; m2++) {
                                packed[base + m2] = vals[2 * m2] | (vals[2 * m2 + 1] << 4);
                            }
                        }
                    }
                    rec = { wdt : 'uint4', w : packed, scales : scales, mins : mins,
                            zp8 : false, p : ptr, pe : ptr + nbytes };
                    S.qcache.set(key, rec);
                }
                const wq = b.constant({ dataType : 'uint4', shape : [n, k], dimensions : [n, k] }, rec.w.slice());
                const sc = b.constant({ dataType : 'float32', shape : [n, nb], dimensions : [n, nb] }, rec.scales.slice());
                const zp = b.constant({ dataType : 'uint4', shape : [n, nb], dimensions : [n, nb] },
                                      new Uint8Array((n * nb + 1) >> 1));
                let out = b.dequantizeLinear(wq, sc, zp);
                const mn = b.constant({ dataType : 'float32', shape : [n, nb, 1], dimensions : [n, nb, 1] }, rec.mins.slice());
                out = b.reshape(out, [n, nb, 32]);
                out = b.sub(out, mn);
                return b.reshape(out, [n, k]);
            }
            if (!rec) {
                const raw = HEAPU8.subarray(ptr, ptr + nbytes);
                const scales = new Float32Array(n * nb);
                if (q === 'q4_0') {
                    const packed = new Uint8Array(n * k / 2); /* webnn uint4: adjacent pairs, low first */
                    for (let r = 0; r < n; r++) {
                        for (let bl = 0; bl < nb; bl++) {
                            const off = (r * nb + bl) * 18;
                            scales[r * nb + bl] = S.f16(raw[off] | (raw[off + 1] << 8));
                            const base = (r * k + bl * 32) >> 1;
                            for (let j = 0; j < 8; j++) {
                                const b0 = raw[off + 2 + 2*j];
                                const b1 = raw[off + 3 + 2*j];
                                packed[base + j]     = (b0 & 0xF) | ((b1 & 0xF) << 4);
                                packed[base + 8 + j] = (b0 >> 4) | (b1 & 0xF0);
                            }
                        }
                    }
                    rec = { wdt : 'uint4', w : packed, scales : scales, zp8 : true, p : ptr, pe : ptr + nbytes };
                } else {
                    const w = new Int8Array(n * k);
                    const i8 = new Int8Array(HEAPU8.buffer, ptr, nbytes);
                    for (let r = 0; r < n; r++) {
                        for (let bl = 0; bl < nb; bl++) {
                            const off = (r * nb + bl) * 34;
                            scales[r * nb + bl] = S.f16(raw[off] | (raw[off + 1] << 8));
                            w.set(i8.subarray(off + 2, off + 34), r * k + bl * 32);
                        }
                    }
                    rec = { wdt : 'int8', w : w, scales : scales, zp8 : false, p : ptr, pe : ptr + nbytes };
                }
                S.qcache.set(key, rec);
            }
            /* constant() may detach its buffer: always hand it a copy, the
               cached arrays are reused across graph builds */
            const wq = b.constant({ dataType : rec.wdt, shape : [n, k], dimensions : [n, k] }, rec.w.slice());
            const sc = b.constant({ dataType : 'float32', shape : [n, nb], dimensions : [n, nb] }, rec.scales.slice());
            let zpData;
            if (rec.zp8) {
                zpData = new Uint8Array((n * nb + 1) >> 1);
                zpData.fill(0x88);
            } else {
                zpData = new Int8Array(n * nb);
            }
            const zp = b.constant({ dataType : rec.wdt, shape : [n, nb], dimensions : [n, nb] }, zpData);
            return b.dequantizeLinear(wq, sc, zp);
        };
        /* persistent flat MLTensor for a device-resident region; host bytes
           uploaded once at creation, afterwards the device copy is authoritative */
        S.getResident = async function(ptr, dt, shape, nbytes) {
            const key = 'r' + ptr;
            let rec = S.resident.get(key);
            if (!rec) {
                rec = {
                    cur : await S.context.createTensor({ dataType : dt, shape : shape, dimensions : shape, readable : true, writable : true }),
                    alt : null, dt : dt, shape : shape, p : ptr,
                };
                S.context.writeTensor(rec.cur, HEAPU8.subarray(ptr, ptr + nbytes));
                S.resident.set(key, rec);
            }
            return rec;
        };
        /* compile the multi-op graph described by descStr:
           { f16, ext:[{dt,shape}...], nodes:[{op,dt,shape,src,ss,sl?,...params}...], outs:[...] }
           src ref < 0 means external input -(ref+1), otherwise an earlier node index.
           ss[k] is the shape the consumer wants (reshape applied when it differs),
           sl[k] is an optional element offset (1D slice of the producer). */
        S.buildGraph = async function(descStr, inT, nIn) {
            const d = JSON.parse(descStr);
            const b = new MLGraphBuilder(S.context);
            const constRanges = [];
            const ext = d.ext.map(function(e, i) {
                if (e.q) {
                    /* quantized weights baked as dequantized graph constants */
                    const ptr = HEAPU32[inT + i*3];
                    const nb = HEAPU32[inT + i*3 + 1];
                    constRanges.push([ptr, ptr + nb]);
                    return S.dequantConst(b, e.q, e.shape, ptr, nb);
                }
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
                    /* strided view: flatten, slice the dense span, reshape to the
                       container, slice the view extents, permute to logical order */
                    const cs = n.cv[k];
                    const span = nel(cs);
                    if (sl !== null && sl !== undefined) {
                        op = b.reshape(op, [nel(have)]);
                        op = b.slice(op, [sl], [span]);
                    } else if (have.length !== 1 || have[0] !== span) {
                        op = b.reshape(op, [span]);
                    }
                    op = b.reshape(op, cs);
                    const vshape = [0, 0, 0, 0];
                    for (let j = 0; j < 4; j++) {
                        vshape[vp[j]] = want[j];
                    }
                    if (!same(cs, vshape)) {
                        op = b.slice(op, [0, 0, 0, 0], vshape);
                    }
                    op = b.transpose(op, { permutation : vp });
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
                    case 'set_rows':
                    {
                        /* KV-cache write: dst[idx[i], :] = src[i, :] via scatterND.
                           src[2] is the cache region [N, D], src[1] the row indices */
                        const cache = srcOp(n, 2, null);
                        const idx = b.reshape(srcOp(n, 1, null), [n.ss[1][0], 1]);
                        let upd = srcOp(n, 0, null);
                        if (srcDt(n, 0) !== n.dt) {
                            upd = b.cast(upd, n.dt);
                        }
                        out = b.scatterND(cache, idx, upd);
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
            /* d.outs is [[node index, resident 0|1, chained 0|1], ...]; resident
               outputs ping-pong the persistent region tensor; chained outputs
               bind a pooled boundary tensor - both resolved at dispatch time */
            const outputsObj = {};
            for (const o of d.outs) {
                let op = ops[o[0]];
                if (o[1]) {
                    op = b.reshape(op, [nel(d.nodes[o[0]].shape)]); /* resident regions are flat */
                }
                outputsObj['n' + o[0]] = op;
            }
            const graph = await b.build(outputsObj);
            console.log('ggml-webnn: compiled graph #' + (S.graphs.size + 1) + ' (' + d.nodes.length + ' nodes)');
            const outInfo = [];
            for (const o of d.outs) {
                const n = d.nodes[o[0]];
                const info = { name : 'n' + o[0], resident : !!o[1], chain : !!o[2],
                               dt : n.dt, shape : n.shape, tensor : null };
                if (!info.resident && !info.chain) {
                    info.tensor = await S.context.createTensor(
                        { dataType : n.dt, shape : n.shape, dimensions : n.shape, readable : true });
                }
                outInfo.push(info);
            }
            const entry = {
                graph : graph, ext : d.ext, outInfo : outInfo, constRanges : constRanges,
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

// whether the WebNN implementation exposes scatterND (needed for SET_ROWS)
EM_JS(int, ggml_webnn_js_has_scatter, (), {
    return globalThis.__ggml_webnn && globalThis.__ggml_webnn.canScatter ? 1 : 0;
});

// whether the embedding page requests output pruning (GGML_WEBNN_PRUNE)
EM_JS(int, ggml_webnn_js_prune, (), {
    return globalThis.GGML_WEBNN_PRUNE ? 1 : 0;
});

// max nodes per compiled graph (GGML_WEBNN_CHUNK, 0 = whole split)
EM_JS(int, ggml_webnn_js_chunk, (), {
    return globalThis.GGML_WEBNN_CHUNK ? Number(globalThis.GGML_WEBNN_CHUNK) : 0;
});

// hybrid phase routing: only accept heavy ops whose token-batch size is in
// [GGML_WEBNN_MIN_BATCH, GGML_WEBNN_MAX_BATCH] (0 = unbounded)
EM_JS(int, ggml_webnn_js_min_batch, (), {
    return globalThis.GGML_WEBNN_MIN_BATCH ? Number(globalThis.GGML_WEBNN_MIN_BATCH) : 0;
});
EM_JS(int, ggml_webnn_js_max_batch, (), {
    return globalThis.GGML_WEBNN_MAX_BATCH ? Number(globalThis.GGML_WEBNN_MAX_BATCH) : 0;
});

// drop cached device tensors whose host region overlaps [ptr, ptr+size):
// called when host memory is freed, cleared or overwritten
EM_JS(void, ggml_webnn_js_invalidate, (void * ptr, size_t size), {
    const S = globalThis.__ggml_webnn;
    if (!S) {
        return;
    }
    const lo = Number(ptr);
    const hi = lo + Number(size);
    for (const m of [S.tpool, S.resident, S.chained]) {
        for (const e of Array.from(m.entries())) {
            const rec = e[1];
            if (rec.p >= lo && rec.p < hi) {
                if (rec.t) { rec.t.destroy(); }
                if (rec.cur) { rec.cur.destroy(); }
                if (rec.alt) { rec.alt.destroy(); }
                m.delete(e[0]);
            }
        }
    }
    /* drop repack caches and graphs whose baked constants overlap the range */
    if (S.qcache) {
        for (const e of Array.from(S.qcache.entries())) {
            if (e[1].p < hi && e[1].pe > lo) {
                S.qcache.delete(e[0]);
            }
        }
    }
    for (const e of Array.from(S.graphs.entries())) {
        const cr = e[1].constRanges || [];
        for (const r of cr) {
            if (r[0] < hi && r[1] > lo) {
                S.graphs.delete(e[0]);
                break;
            }
        }
    }
});

// begin a graph_compute call: flush any leftovers and open a new chain epoch
EM_ASYNC_JS(int, ggml_webnn_js_begin, (), {
    const S = globalThis.__ggml_webnn;
    try {
        if (S.pending.length) {
            await S.flush();
        }
        S.chainEpoch++;
        return 0;
    } catch (e) {
        console.error('ggml-webnn: begin failed:', e);
        return 1;
    }
});

// await all enqueued readbacks and write them to host memory
EM_ASYNC_JS(int, ggml_webnn_js_flush, (), {
    const S = globalThis.__ggml_webnn;
    try {
        await S.flush();
        return 0;
    } catch (e) {
        console.error('ggml-webnn: flush failed:', e);
        return 1;
    }
});

// ENQUEUE one compiled graph: upload changed inputs, dispatch (fire-and-forget)
// and queue readbacks - they complete at the next flush. in_tab is n_in x
// [ptr, nbytes, flags] (bit0 weight, bit1 resident, bit2 i64->i32, bit3 const);
// out_tab is n_out x [ptr, nbytes, flags] (bit0 resident, bit1 readback,
// bit2 chained: result stays on device for a later segment this epoch).
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
            entry = await S.buildGraph(descStr, inT, n_in);
        }
        const same = function(a, c) { return a.length === c.length && a.every(function(v, i) { return v === c[i]; }); };

        const feeds = {};
        for (let i = 0; i < n_in; i++) {
            const ptr = HEAPU32[inT + i*3];
            const nb = HEAPU32[inT + i*3 + 1];
            const fl = HEAPU32[inT + i*3 + 2];
            const e = entry.ext[i];
            if (fl & 8) {
                continue; /* baked as graph constants at build time */
            }
            if (fl & 2) {
                /* device-resident region (KV cache): bind the persistent tensor */
                const rec = await S.getResident(ptr, e.dt, e.shape, nb);
                feeds['x' + i] = rec.cur;
                continue;
            }
            /* produced on-device by an earlier segment this epoch? bind directly */
            const ch = S.chained.get('c' + ptr);
            if (ch && ch.epoch === S.chainEpoch && ch.dt === e.dt && same(ch.shape, e.shape)) {
                feeds['x' + i] = ch.t;
                continue;
            }
            /* the host copy is the source: if its readback is still in flight,
               flush first so we do not upload stale data */
            if (S.pendingPtrs.has(ptr)) {
                await S.flush();
            }
            const key = ptr + '|' + entry.inKeys[i];
            let rec = S.tpool.get(key);
            if (!rec) {
                rec = { t : await S.context.createTensor(
                            { dataType : e.dt, shape : e.shape, dimensions : e.shape, writable : true }),
                        written : false, p : ptr };
                S.tpool.set(key, rec);
            }
            const isWeight = (fl & 1) !== 0;
            if (!isWeight || !rec.written) {
                if (fl & 4) {
                    /* int64 host data feeding an int32 tensor (scatter indices) */
                    const n64 = nb >> 3;
                    const src = new BigInt64Array(HEAPU8.buffer, ptr, n64);
                    const conv = new Int32Array(n64);
                    for (let k = 0; k < n64; k++) {
                        conv[k] = Number(src[k]);
                    }
                    S.context.writeTensor(rec.t, conv);
                } else {
                    /* writeTensor copies synchronously, so a heap view is safe here */
                    S.context.writeTensor(rec.t, HEAPU8.subarray(ptr, ptr + nb));
                }
                rec.written = true;
            }
            feeds['x' + i] = rec.t;
        }
        const inputTensors = new Set(Object.values(feeds));

        /* outputs: resident regions ping-pong cur/alt; chained results bind a
           pooled tensor reused as a later segment's input; the rest use
           per-graph tensors */
        const outFeeds = {};
        const swaps = [];
        for (let j = 0; j < n_out; j++) {
            const info = entry.outInfo[j];
            const ptr = HEAPU32[outT + j*3];
            if (info.resident) {
                /* the scatter graph always reads the region, so the record exists */
                const rec = S.resident.get('r' + ptr);
                if (!rec.alt) {
                    rec.alt = await S.context.createTensor(
                        { dataType : rec.dt, shape : rec.shape, dimensions : rec.shape, readable : true, writable : true });
                }
                outFeeds[info.name] = rec.alt;
                swaps.push(rec);
            } else if (info.chain) {
                let rec = S.chained.get('c' + ptr);
                const stale = !rec || rec.dt !== info.dt || !same(rec.shape, info.shape) ||
                              inputTensors.has(rec.t); /* cannot read and write one tensor */
                if (stale) {
                    rec = { t : await S.context.createTensor(
                                { dataType : info.dt, shape : info.shape, dimensions : info.shape, readable : true, writable : true }),
                            dt : info.dt, shape : info.shape, p : ptr, epoch : 0 };
                    S.chained.set('c' + ptr, rec);
                }
                rec.epoch = S.chainEpoch;
                outFeeds[info.name] = rec.t;
            } else {
                outFeeds[info.name] = info.tensor;
            }
        }

        S.context.dispatch(entry.graph, feeds, outFeeds);

        for (const rec of swaps) {
            const t = rec.cur;
            rec.cur = rec.alt;
            rec.alt = t;
        }

        /* queue readbacks for host-materialized outputs; flushed in order later */
        for (let j = 0; j < n_out; j++) {
            const fl = HEAPU32[outT + j*3 + 2];
            if (!(fl & 2)) {
                continue;
            }
            const info = entry.outInfo[j];
            const ptr = HEAPU32[outT + j*3];
            let t;
            if (info.resident) {
                t = S.resident.get('r' + ptr).cur;
            } else if (info.chain) {
                t = S.chained.get('c' + ptr).t;
            } else {
                t = info.tensor;
            }
            S.pending.push({ promise : S.context.readTensor(t), ptr : ptr });
            S.pendingPtrs.add(ptr);
        }
        return 0;
    } catch (e) {
        console.error('ggml-webnn: graph dispatch failed:', e, UTF8ToString(Number(desc)).slice(0, 4000));
        return 1;
    }
});

static bool   g_webnn_available   = false;
static bool   g_webnn_force_f16   = false;
static bool   g_webnn_has_scatter = false;
static bool   g_webnn_prune       = false; // skip host writeback of consumed intermediates
static int    g_webnn_chunk       = 0;     // max nodes per compiled graph (0 = whole split)
static int    g_webnn_min_batch   = 0;     // hybrid routing: heavy-op token-batch window
static int    g_webnn_max_batch   = 0;
static size_t g_webnn_n_ops       = 0; // ggml nodes executed via WebNN
static size_t g_webnn_n_dispatch  = 0; // compiled-graph dispatches
static std::map<std::string, size_t> g_webnn_op_tally;  // per-op execution counts
static std::map<const void *, size_t> g_webnn_resident; // device-resident regions (KV caches): base ptr -> bytes

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

// quantized weight types translated to dequantizeLinear graph constants
static bool ggml_webnn_is_quant(ggml_type t) {
    return t == GGML_TYPE_Q4_0 || t == GGML_TYPE_Q8_0 || t == GGML_TYPE_Q4_K;
}

// quantized tensors must be immutable 2D weights with whole blocks per row
static bool ggml_webnn_quant_ok(const ggml_tensor * t) {
    if (!ggml_webnn_is_quant(t->type) || !ggml_is_contiguous(t) ||
        t->ne[1] <= 0 || t->ne[2] != 1 || t->ne[3] != 1) {
        return false;
    }
    return t->ne[0] % (t->type == GGML_TYPE_Q4_K ? 256 : 32) == 0;
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

// check whether t is a (possibly permuted) strided slice of a dense block
// starting at t->data: covers ggml_permute views, KV cache head-interleaved
// views, and padded views like the transposed V cache ([n_kv, ...] rows of an
// n_ctx-wide container). fills:
//   perm_out:      JS transpose permutation restoring logical order
//   container_out: dense container shape in memory order (outermost first)
//   span_out:      container extent in elements from t->data
// the JS translation is: flat[span] -> reshape(container) -> slice(view sizes)
// -> transpose(perm).
static bool ggml_webnn_strided_view(const ggml_tensor * t, int perm_out[GGML_MAX_DIMS],
                                    int64_t container_out[GGML_MAX_DIMS], int64_t & span_out) {
    const size_t ts = ggml_type_size(t->type);
    if (ts == 0 || ggml_blck_size(t->type) != 1) {
        return false;
    }
    // dims with ne>1, ordered by stride ascending
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
    // innermost run must be element-contiguous; each level must divide the next
    int64_t cap[GGML_MAX_DIMS]; // container capacity per (ne>1) dim, ascending-stride order
    if (n > 0) {
        if (t->nb[dims[0]] != ts) {
            return false;
        }
        for (int a = 0; a + 1 < n; a++) {
            if (t->nb[dims[a + 1]] % t->nb[dims[a]] != 0) {
                return false;
            }
            cap[a] = (int64_t) (t->nb[dims[a + 1]] / t->nb[dims[a]]);
            if (cap[a] < t->ne[dims[a]]) {
                return false; // overlapping strides
            }
        }
        cap[n - 1] = t->ne[dims[n - 1]]; // outermost is unbounded
        span_out = (int64_t) (t->ne[dims[n - 1]] * t->nb[dims[n - 1]] / ts);
    } else {
        span_out = 1;
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
    // container shape follows memory order
    for (int k = 0; k < GGML_MAX_DIMS; k++) {
        container_out[k] = 1;
        for (int a = 0; a < n; a++) {
            if (dims[a] == o[k]) {
                container_out[k] = cap[a];
                break;
            }
        }
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

// contiguous, or a strided view we can translate (reshape+slice+transpose)
static bool ggml_webnn_view_ok(const ggml_tensor * t) {
    if (ggml_is_contiguous(t)) {
        return true;
    }
    int perm[GGML_MAX_DIMS];
    int64_t container[GGML_MAX_DIMS];
    int64_t span;
    return ggml_webnn_strided_view(t, perm, container, span);
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
        case GGML_OP_SET_ROWS: fields = "\"op\":\"set_rows\""; return true;
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
    std::vector<ggml_tensor *> all_nodes;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if (ggml_webnn_is_noop(node) || ggml_is_empty(node) || (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        all_nodes.push_back(node);
    }
    if (all_nodes.empty()) {
        return GGML_STATUS_SUCCESS;
    }

    // 1a. partition into segments of at most GGML_WEBNN_CHUNK nodes, each
    // compiled and dispatched as its own graph. tensors consumed by a LATER
    // segment are either chained device-side (when every cross-segment
    // consumer reads the producer tensor directly, same shape/type) or
    // materialized to host before the consumer uploads
    const size_t chunk_sz = g_webnn_chunk > 0 ? (size_t) g_webnn_chunk : all_nodes.size();
    std::vector<bool> cross_seg(all_nodes.size(), false);
    std::vector<bool> chain_ok(all_nodes.size(), true);
    if (chunk_sz < all_nodes.size()) {
        std::map<const ggml_tensor *, size_t> global_idx;
        for (size_t i = 0; i < all_nodes.size(); i++) {
            global_idx[all_nodes[i]] = i;
        }
        for (size_t i = 0; i < all_nodes.size(); i++) {
            for (int k = 0; k < GGML_MAX_SRC && all_nodes[i]->src[k]; k++) {
                const ggml_tensor * s0 = all_nodes[i]->src[k];
                const ggml_tensor * s = s0;
                while (global_idx.find(s) == global_idx.end() && s->view_src) {
                    s = s->view_src;
                }
                auto it = global_idx.find(s);
                if (it != global_idx.end() && it->second / chunk_sz != i / chunk_sz) {
                    cross_seg[it->second] = true;
                    if (s0 != s) {
                        chain_ok[it->second] = false; // consumed through a view: shapes differ
                    }
                }
            }
        }
    }

    if (ggml_webnn_js_begin() != 0) {
        return GGML_STATUS_FAILED;
    }

    for (size_t seg_base = 0; seg_base < all_nodes.size(); seg_base += chunk_sz) {

    const std::vector<ggml_tensor *> nodes(all_nodes.begin() + seg_base,
        all_nodes.begin() + std::min(seg_base + chunk_sz, all_nodes.size()));
    std::map<const ggml_tensor *, int> node_idx;
    for (size_t i = 0; i < nodes.size(); i++) {
        node_idx[nodes[i]] = (int) i;
    }

    // 1b. device-resident regions: register the root of every SET_ROWS
    // destination (the KV caches); remember the last scatter per root,
    // which carries the region update out of this dispatch
    std::map<const void *, int> last_scatter;
    for (size_t i = 0; i < nodes.size(); i++) {
        if (nodes[i]->op != GGML_OP_SET_ROWS) {
            continue;
        }
        const ggml_tensor * root = nodes[i];
        while (root->view_src) {
            root = root->view_src;
        }
        if (root->type != nodes[i]->type || root->data != nodes[i]->data ||
            ggml_nelements(root) != ggml_nelements(nodes[i])) {
            GGML_LOG_ERROR("ggml-webnn: SET_ROWS dst is not a full-span view of its root\n");
            return GGML_STATUS_FAILED;
        }
        g_webnn_resident[root->data] = ggml_nbytes(root);
        last_scatter[root->data] = (int) i;
    }

    // 2. translate into a graph descriptor (doubles as the compile cache key)
    std::vector<std::string> ext_descs;
    std::vector<uint32_t>    in_tab; // [ptr, nbytes, flags] per external input
    std::map<std::string, int> ext_idx;
    std::vector<bool> consumed(nodes.size(), false); // consumed within this split
    std::map<const void *, int> scattered; // resident root ptr -> latest SET_ROWS node so far

    // resolve a src tensor to an operand reference:
    //   ref < 0  -> external input -(ref+1)  (data read from host at dispatch)
    //   ref >= 0 -> node index in this split (+ optional 1D element slice)
    // non-contiguous (permuted-dense) views additionally get a JS transpose
    // permutation in vp_json; their external inputs are declared flat
    auto resolve = [&](const ggml_tensor * t, int rank, int & ref, int64_t & slice_off,
                       std::string & vp_json, std::string & cs_json) -> bool {
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
        cs_json = "null";

        const bool contig = ggml_is_contiguous(t);
        int64_t span = ggml_nelements(t);
        if (!contig) {
            int perm[GGML_MAX_DIMS];
            int64_t container[GGML_MAX_DIMS];
            if (rank != 4 || !ggml_webnn_strided_view(t, perm, container, span)) {
                return false;
            }
            vp_json = "[" + std::to_string(perm[0]) + "," + std::to_string(perm[1]) + "," +
                      std::to_string(perm[2]) + "," + std::to_string(perm[3]) + "]";
            cs_json = "[" + std::to_string(container[0]) + "," + std::to_string(container[1]) + "," +
                      std::to_string(container[2]) + "," + std::to_string(container[3]) + "]";
        }

        // when the owner is the chain root and that region was scattered into by
        // an earlier SET_ROWS of this split, the scatter node is the producer
        if (it == node_idx.end() && owner->view_src == nullptr) {
            auto sit = scattered.find(owner->data);
            if (sit != scattered.end()) {
                it = node_idx.find(nodes[sit->second]);
            }
        }

        if (it != node_idx.end()) {
            const ggml_tensor * producer = it->first;
            if (producer->type != t->type) {
                return false; // type-punning views are not supported
            }
            ref = it->second;
            consumed[ref] = true;
            const size_t off_bytes = (const char *) t->data - (const char *) producer->data;
            if (off_bytes % ggml_type_size(t->type) != 0) {
                return false;
            }
            if (!contig || off_bytes != 0 || ggml_nelements(t) != ggml_nelements(producer)) {
                slice_off = (int64_t) (off_bytes / ggml_type_size(t->type));
            }
            return true;
        }

        // device-resident region (KV cache): bind the full flat region as input;
        // the view becomes a slice of it
        if (owner->view_src == nullptr) {
            auto rit = g_webnn_resident.find(owner->data);
            if (rit != g_webnn_resident.end()) {
                if (owner->type != t->type) {
                    return false;
                }
                const char * dt = ggml_webnn_dt(t->type);
                if (dt == nullptr) {
                    return false;
                }
                const int64_t root_nel = (int64_t) (rit->second / ggml_type_size(t->type));
                const size_t off_bytes = (const char *) t->data - (const char *) owner->data;
                if (off_bytes % ggml_type_size(t->type) != 0) {
                    return false;
                }
                if (!contig || off_bytes != 0 || ggml_nelements(t) != root_nel) {
                    slice_off = (int64_t) (off_bytes / ggml_type_size(t->type));
                }
                const std::string shape = "[" + std::to_string(root_nel) + "]";
                char key[64];
                snprintf(key, sizeof(key), "%p|%s|", owner->data, dt);
                const std::string k = key + shape;
                auto eit = ext_idx.find(k);
                int idx;
                if (eit != ext_idx.end()) {
                    idx = eit->second;
                } else {
                    idx = (int) ext_descs.size();
                    ext_idx[k] = idx;
                    ext_descs.push_back("{\"dt\":\"" + std::string(dt) + "\",\"shape\":" + shape + "}");
                    in_tab.push_back((uint32_t) (uintptr_t) owner->data);
                    in_tab.push_back((uint32_t) rit->second);
                    in_tab.push_back(2); // resident
                }
                ref = -(idx + 1);
                return true;
            }
        }

        // quantized weights: repacked into dequantizeLinear graph constants at
        // build time (data read from the table pointer); logical shape [n, k]
        if (ggml_webnn_is_quant(t->type)) {
            if (!ggml_webnn_quant_ok(t)) {
                return false;
            }
            const char * qn = t->type == GGML_TYPE_Q4_0 ? "q4_0" :
                              t->type == GGML_TYPE_Q4_K ? "q4_k" : "q8_0";
            const std::string shape = "[" + std::to_string(t->ne[1]) + "," + std::to_string(t->ne[0]) + "]";
            char key[64];
            snprintf(key, sizeof(key), "%p|%s|", t->data, qn);
            const std::string k = key + shape;
            auto eit = ext_idx.find(k);
            int idx;
            if (eit != ext_idx.end()) {
                idx = eit->second;
            } else {
                idx = (int) ext_descs.size();
                ext_idx[k] = idx;
                ext_descs.push_back("{\"dt\":\"float32\",\"shape\":" + shape + ",\"q\":\"" + qn +
                                    "\",\"cp\":" + std::to_string((uintptr_t) t->data) + "}");
                in_tab.push_back((uint32_t) (uintptr_t) t->data);
                in_tab.push_back((uint32_t) ggml_nbytes(t));
                in_tab.push_back(8); // baked constant
            }
            ref = -(idx + 1);
            return true;
        }

        // external input (leaf, weight, or a tensor computed outside this split);
        // strided views are declared as their flat dense span, i64 indices as i32
        const bool cvt_i64 = t->type == GGML_TYPE_I64;
        const char * dt = cvt_i64 ? "int32" : ggml_webnn_dt(t->type);
        if (dt == nullptr) {
            return false;
        }
        const size_t ts = cvt_i64 ? sizeof(int64_t) : ggml_type_size(t->type);
        const std::string shape = contig ? ggml_webnn_shape_js(t, rank)
                                         : "[" + std::to_string(span) + "]";
        const size_t nbytes = contig ? ggml_nbytes(t)
                                     : (size_t) span * ts;
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
            if (cvt_i64) {
                flags |= 4;
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

        const int node_rank = (node->op == GGML_OP_GET_ROWS || node->op == GGML_OP_SET_ROWS) ? 2 : 4;

        // CPY carries its destination alias as src[1] - it is not a data input
        const int n_srcs = node->op == GGML_OP_CPY ? 1 : GGML_MAX_SRC;

        std::string src_json = "[";
        std::string ss_json  = "[";
        std::string sl_json  = "[";
        std::string vp_json  = "[";
        std::string cv_json  = "[";
        for (int k = 0; k < n_srcs && node->src[k]; k++) {
            // per-src rank: GET_ROWS/SET_ROWS use rank-2 data + rank-1 indices,
            // ROPE positions are a rank-1 i32 vector
            int rank = 4;
            if (node->op == GGML_OP_GET_ROWS || node->op == GGML_OP_SET_ROWS) {
                rank = k == 1 ? 1 : 2;
            } else if (node->op == GGML_OP_ROPE && k == 1) {
                rank = 1;
            }
            int ref;
            int64_t slice_off;
            std::string vp;
            std::string cs;
            if (!resolve(node->src[k], rank, ref, slice_off, vp, cs)) {
                GGML_LOG_ERROR("ggml-webnn: cannot resolve src %d of %s\n", k, ggml_op_desc(node));
                return GGML_STATUS_FAILED;
            }
            if (k > 0) {
                src_json += ",";
                ss_json += ",";
                sl_json += ",";
                vp_json += ",";
                cv_json += ",";
            }
            src_json += std::to_string(ref);
            ss_json  += ggml_webnn_shape_js(node->src[k], rank);
            sl_json  += slice_off < 0 ? "null" : std::to_string(slice_off);
            vp_json  += vp;
            cv_json  += cs;
        }
        src_json += "]";
        ss_json += "]";
        sl_json += "]";
        vp_json += "]";
        cv_json += "]";

        if (i > 0) {
            desc += ",";
        }
        desc += "{" + fields +
                ",\"dt\":\"" + ggml_webnn_dt(node->type) + "\"" +
                ",\"shape\":" + ggml_webnn_shape_js(node, node_rank) +
                ",\"src\":" + src_json +
                ",\"ss\":" + ss_json +
                ",\"sl\":" + sl_json +
                ",\"vp\":" + vp_json +
                ",\"cv\":" + cv_json + "}";

        if (node->op == GGML_OP_SET_ROWS) {
            scattered[node->data] = (int) i; // later reads of this region use the scatter result
        }
    }

    // 3. materialize results back to host. resident scatters ping-pong the
    // persistent tensor instead of reading back. by default every other node
    // is materialized (sound: the scheduler may hand any tensor to a later
    // split). with GGML_WEBNN_PRUNE only sinks and flagged outputs are read
    // back - valid when the whole graph compiles to a single split.
    desc += "],\"outs\":[";
    std::vector<uint32_t> out_tab; // [ptr, nbytes, flags] per output
    int n_out = 0;
    for (size_t i = 0; i < nodes.size(); i++) {
        const ggml_tensor * node = nodes[i];
        bool is_res = false;
        if (node->op == GGML_OP_SET_ROWS) {
            auto lit = last_scatter.find(node->data);
            is_res = lit != last_scatter.end() && lit->second == (int) i;
        }
        const bool is_chain = !is_res && cross_seg[seg_base + i] && chain_ok[seg_base + i];
        bool readback = !consumed[i] || (node->flags & GGML_TENSOR_FLAG_OUTPUT) ||
                        (cross_seg[seg_base + i] && !is_chain);
        if (!g_webnn_prune && !is_res) {
            readback = true;
        }
        if (!is_res && !is_chain && !readback) {
            continue;
        }
        if (n_out > 0) {
            desc += ",";
        }
        desc += "[" + std::to_string(i) + "," + (is_res ? "1" : "0") + "," + (is_chain ? "1" : "0") + "]";
        out_tab.push_back((uint32_t) (uintptr_t) node->data);
        out_tab.push_back((uint32_t) (is_res ? g_webnn_resident[node->data] : ggml_nbytes(node)));
        out_tab.push_back((is_res ? 1u : 0u) | (readback ? 2u : 0u) | (is_chain ? 4u : 0u));
        n_out++;
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
        out_tab.data(), n_out);

    if (ret != 0) {
        GGML_LOG_ERROR("ggml-webnn: graph dispatch failed (%d nodes)\n", (int) nodes.size());
        return GGML_STATUS_FAILED;
    }

    g_webnn_n_ops += nodes.size();
    g_webnn_n_dispatch++;
    for (const ggml_tensor * node : nodes) {
        g_webnn_op_tally[ggml_op_desc(node)]++;
    }

    } // segment loop

    // all segments are enqueued; one await completes every host readback
    if (ggml_webnn_js_flush() != 0) {
        return GGML_STATUS_FAILED;
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
// buffer type: host memory (CPU-compatible) with invalidation hooks - frees,
// clears and host writes drop the corresponding cached device tensors
//

static void ggml_webnn_invalidate(const void * ptr, size_t size) {
    ggml_webnn_js_invalidate(const_cast<void *>(ptr), size);
    for (auto it = g_webnn_resident.begin(); it != g_webnn_resident.end();) {
        const char * p = (const char *) it->first;
        if (p >= (const char *) ptr && p < (const char *) ptr + size) {
            it = g_webnn_resident.erase(it);
        } else {
            ++it;
        }
    }
}

struct ggml_webnn_buffer_ctx {
    void * base;
    size_t size;
};

static void ggml_backend_webnn_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_webnn_buffer_ctx * ctx = (ggml_webnn_buffer_ctx *) buffer->context;
    ggml_webnn_invalidate(ctx->base, ctx->size);
    free(ctx->base);
    delete ctx;
}

static void * ggml_backend_webnn_buffer_get_base(ggml_backend_buffer_t buffer) {
    return ((ggml_webnn_buffer_ctx *) buffer->context)->base;
}

static void ggml_backend_webnn_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_webnn_invalidate((const char *) tensor->data + offset, size);
    memset((char *) tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_webnn_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_webnn_invalidate((const char *) tensor->data + offset, size);
    memcpy((char *) tensor->data + offset, data, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_webnn_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    memcpy(data, (const char *) tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

static bool ggml_backend_webnn_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_backend_buffer_is_host(src->buffer)) {
        ggml_webnn_invalidate(dst->data, ggml_nbytes(dst));
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_webnn_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_webnn_buffer_ctx * ctx = (ggml_webnn_buffer_ctx *) buffer->context;
    ggml_webnn_invalidate(ctx->base, ctx->size);
    memset(ctx->base, value, ctx->size);
}

static const struct ggml_backend_buffer_i ggml_backend_webnn_buffer_i = {
    /* .free_buffer   = */ ggml_backend_webnn_buffer_free_buffer,
    /* .get_base      = */ ggml_backend_webnn_buffer_get_base,
    /* .init_tensor   = */ NULL,
    /* .memset_tensor = */ ggml_backend_webnn_buffer_memset_tensor,
    /* .set_tensor    = */ ggml_backend_webnn_buffer_set_tensor,
    /* .get_tensor    = */ ggml_backend_webnn_buffer_get_tensor,
    /* .cpy_tensor    = */ ggml_backend_webnn_buffer_cpy_tensor,
    /* .clear         = */ ggml_backend_webnn_buffer_clear,
    /* .reset         = */ NULL,
};

static const char * ggml_backend_webnn_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "WebNN_Host";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_webnn_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * base = nullptr;
    if (posix_memalign(&base, 64, size > 0 ? size : 64) != 0) {
        return nullptr;
    }
    ggml_webnn_buffer_ctx * ctx = new ggml_webnn_buffer_ctx { base, size };
    return ggml_backend_buffer_init(buft, ggml_backend_webnn_buffer_i, ctx, size);
}

static size_t ggml_backend_webnn_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 64;

    GGML_UNUSED(buft);
}

static bool ggml_backend_webnn_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_webnn_buffer_type(ggml_backend_dev_t dev) {
    static struct ggml_backend_buffer_type buft = {
        /* .iface = */ {
            /* .get_name       = */ ggml_backend_webnn_buffer_type_get_name,
            /* .alloc_buffer   = */ ggml_backend_webnn_buffer_type_alloc_buffer,
            /* .get_alignment  = */ ggml_backend_webnn_buffer_type_get_alignment,
            /* .get_max_size   = */ NULL,
            /* .get_alloc_size = */ NULL,
            /* .is_host        = */ ggml_backend_webnn_buffer_type_is_host,
        },
        /* .device  = */ nullptr,
        /* .context = */ nullptr,
    };
    buft.device = dev;
    return &buft;
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
    return ggml_backend_webnn_buffer_type(dev);
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

// hybrid phase routing: heavy ops outside the configured token-batch window
// are declined so the scheduler places them on another backend (e.g. WebGPU)
static bool ggml_webnn_batch_ok(int64_t batch) {
    if (g_webnn_min_batch > 0 && batch < g_webnn_min_batch) {
        return false;
    }
    if (g_webnn_max_batch > 0 && batch > g_webnn_max_batch) {
        return false;
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
            if (!ggml_webnn_batch_ok(src1->ne[1])) {
                return false;
            }
            // f32/f16 srcs in any combination (strided views allowed), f32 dst
            if (op->type != GGML_TYPE_F32 || !ggml_is_contiguous(op)) {
                return false;
            }
            const bool src0_ok = (ggml_webnn_is_float(src0->type) && ggml_webnn_view_ok(src0)) ||
                                 ggml_webnn_quant_ok(src0);
            if (!src0_ok || !ggml_webnn_is_float(src1->type) || !ggml_webnn_view_ok(src1)) {
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
            if (!ggml_webnn_batch_ok(src0->ne[1])) {
                return false;
            }
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
            // a mask is required: rare mask-less shapes mis-execute on some
            // delegates (observed: hsk=128 GQA returning zeros on LiteRT),
            // and llama always provides one
            if (mask == nullptr) {
                return false;
            }
            if (mask->type != GGML_TYPE_F16 || !ggml_is_contiguous(mask)) {
                return false;
            }
            if (mask->ne[0] != src1->ne[1] || mask->ne[1] != src0->ne[1] || mask->ne[2] != 1 ||
                (mask->ne[3] != src0->ne[3] && mask->ne[3] != 1)) {
                return false;
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
            // float<->float copies (with cast), or same-type i32; strided
            // sources (e.g. CONT of a permuted KV cache view) translate via
            // the strided-view path
            if (!ggml_webnn_view_ok(src0) || !ggml_is_contiguous(op)) {
                return false;
            }
            if (ggml_webnn_is_float(src0->type) && ggml_webnn_is_float(op->type)) {
                return true;
            }
            return src0->type == GGML_TYPE_I32 && op->type == GGML_TYPE_I32;

        case GGML_OP_GET_ROWS:
            return op->type == GGML_TYPE_F32 &&
                   ((ggml_webnn_is_float(src0->type) && ggml_is_contiguous(src0)) || ggml_webnn_quant_ok(src0)) &&
                   src1->type == GGML_TYPE_I32 && ggml_is_contiguous(src1) &&
                   src0->ne[2] == 1 && src0->ne[3] == 1 &&
                   src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1;

        case GGML_OP_SET_ROWS:
        {
            // scatter rows into a device-resident region: f32 rows into an
            // f32/f16 destination that is a full-span, offset-0 view of its
            // root tensor (the KV cache layout)
            if (!g_webnn_has_scatter) {
                return false;
            }
            const ggml_tensor * idx = src1;
            if (!ggml_webnn_is_float(op->type) || src0->type != GGML_TYPE_F32 ||
                (idx->type != GGML_TYPE_I64 && idx->type != GGML_TYPE_I32)) {
                return false;
            }
            if (!ggml_is_contiguous(op) || !ggml_is_contiguous(src0) || !ggml_is_contiguous(idx)) {
                return false;
            }
            if (op->ne[2] != 1 || op->ne[3] != 1 || src0->ne[2] != 1 || src0->ne[3] != 1 ||
                idx->ne[1] != 1 || idx->ne[2] != 1) {
                return false;
            }
            // supports_op may run pre-allocation: compare via view offsets, not data
            size_t off = 0;
            const ggml_tensor * root = op;
            while (root->view_src) {
                off += root->view_offs;
                root = root->view_src;
            }
            return root->type == op->type &&
                   off == 0 &&
                   ggml_nelements(op) == ggml_nelements(root);
        }

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
            g_webnn_has_scatter = ggml_webnn_js_has_scatter() != 0;
            if (!g_webnn_has_scatter) {
                GGML_LOG_WARN("ggml-webnn: scatterND not available, SET_ROWS stays on CPU\n");
            }
            g_webnn_prune = ggml_webnn_js_prune() != 0;
            if (g_webnn_prune) {
                GGML_LOG_INFO("ggml-webnn: output pruning enabled (GGML_WEBNN_PRUNE)\n");
            }
            g_webnn_chunk = ggml_webnn_js_chunk();
            if (g_webnn_chunk > 0) {
                GGML_LOG_INFO("ggml-webnn: compiling graphs in chunks of %d nodes (GGML_WEBNN_CHUNK)\n", g_webnn_chunk);
            }
            g_webnn_min_batch = ggml_webnn_js_min_batch();
            g_webnn_max_batch = ggml_webnn_js_max_batch();
            if (g_webnn_min_batch > 0 || g_webnn_max_batch > 0) {
                GGML_LOG_INFO("ggml-webnn: heavy ops restricted to token batch [%d, %d]\n",
                              g_webnn_min_batch, g_webnn_max_batch ? g_webnn_max_batch : 999999);
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

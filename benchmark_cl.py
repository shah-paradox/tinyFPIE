import argparse
import sys
import time
import os
import cv2
import numpy as np

try:
    import pyopencl as cl
except ImportError:
    print("ERROR: pyopencl not installed. Run: pip install pyopencl")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# OpenCL kernels
# ─────────────────────────────────────────────────────────────────────────────
KERNEL_SOURCE = """
__kernel void kernel_init_r(
    __global const float* tgt,   
    __global const float* grad,  
    __global const int*   mask,  
    __global float*       r,     
    __global float*       p,     
    const int N, const int M)
{
    int px = get_global_id(0);
    int py = get_global_id(1);
    int c  = get_global_id(2);
    if (px >= M || py >= N) return;
    int id  = py * M + px;
    int gid = c * N * M + id;
    if (!mask[id]) {
        r[gid] = 0.0f; p[gid] = 0.0f;
        return;
    }
    float x_c = tgt[gid];
    float x_t = (py > 0   && mask[id - M]) ? tgt[c*N*M + id - M] : 0.0f;
    float x_b = (py < N-1 && mask[id + M]) ? tgt[c*N*M + id + M] : 0.0f;
    float x_l = (px > 0   && mask[id - 1]) ? tgt[c*N*M + id - 1] : 0.0f;
    float x_r = (px < M-1 && mask[id + 1]) ? tgt[c*N*M + id + 1] : 0.0f;
    float Ax = 4.0f * x_c - x_t - x_b - x_l - x_r;
    float res = grad[gid] - Ax;
    r[gid] = res; p[gid] = res;
}

__kernel void kernel_Ap(
    __global const float* p,
    __global const int*   mask,
    __global float*       Ap,
    const int N, const int M)
{
    int px = get_global_id(0);
    int py = get_global_id(1);
    int c  = get_global_id(2);
    if (px >= M || py >= N) return;
    int id  = py * M + px;
    int gid = c * N * M + id;
    if (!mask[id]) {
        Ap[gid] = 0.0f;
        return;
    }
    float p_c = p[gid];
    float p_t = (py > 0   && mask[id - M]) ? p[c*N*M + id - M] : 0.0f;
    float p_b = (py < N-1 && mask[id + M]) ? p[c*N*M + id + M] : 0.0f;
    float p_l = (px > 0   && mask[id - 1]) ? p[c*N*M + id - 1] : 0.0f;
    float p_r = (px < M-1 && mask[id + 1]) ? p[c*N*M + id + 1] : 0.0f;
    Ap[gid] = 4.0f * p_c - (p_t + p_b + p_l + p_r);
}

__kernel void kernel_dot(
    __global const float* a,
    __global const float* b,
    __global const int*   mask,
    __global float*       partials,
    __local  float*       lcache,
    const int N, const int M, const int c)
{
    int px       = get_global_id(0);
    int py       = get_global_id(1);
    int lid      = get_local_id(1) * get_local_size(0) + get_local_id(0);
    int group_id = get_group_id(1) * get_num_groups(0) + get_group_id(0);
    float val = 0.0f;
    if (px < M && py < N) {
        int id  = py * M + px;
        int gid = c * N * M + id;
        if (mask[id]) val = a[gid] * b[gid];
    }
    lcache[lid] = val;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = get_local_size(0) * get_local_size(1) / 2; s > 0; s >>= 1) {
        if (lid < s) lcache[lid] += lcache[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) partials[group_id] = lcache[0];
}

__kernel void kernel_update_xr(
    __global float*       x_vec,
    __global float*       r,
    __global const float* p,
    __global const float* Ap,
    __global const int*   mask,
    const float alpha,
    const int N, const int M, const int c)
{
    int px = get_global_id(0);
    int py = get_global_id(1);
    if (px >= M || py >= N) return;
    int id  = py * M + px;
    int gid = c * N * M + id;
    if (mask[id]) {
        x_vec[gid] += alpha * p[gid];
        r[gid]     -= alpha * Ap[gid];
    }
}

__kernel void kernel_update_p(
    __global float*       p,
    __global const float* r,
    __global const int*   mask,
    const float beta,
    const int N, const int M, const int c)
{
    int px = get_global_id(0);
    int py = get_global_id(1);
    if (px >= M || py >= N) return;
    int id  = py * M + px;
    int gid = c * N * M + id;
    if (mask[id]) {
        p[gid] = r[gid] + beta * p[gid];
    } else {
        p[gid] = 0.0f;
    }
}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing Logic
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(src, mask_img, tgt, mask_on_src, mask_on_tgt):
    """
    Returns:
        mask_bin  : (N, M) int32 binary mask (0/1), border zeroed
        tgt_crop  : (N, M, 3) float32 target region
        grad      : (N, M, 3) float32 RHS gradient with Dirichlet BCs baked in
        tgt_full  : full target array (copy), for final compositing
        (x0, x1, y0, y1) : crop coordinates in tgt
    """
    # Binarise mask
    if mask_img.ndim == 3:
        mask_img = mask_img.mean(-1)
    mask_bin = (mask_img >= 128).astype(np.int32)

    # Zero borders
    mask_bin[0, :] = mask_bin[-1, :] = 0
    mask_bin[:, 0] = mask_bin[:, -1] = 0

    # Tight crop around the active region (±1 px border)
    ys, xs = np.nonzero(mask_bin)
    if len(ys) == 0: return None
    x0, x1 = int(ys.min()) - 1, int(ys.max()) + 2
    y0, y1 = int(xs.min()) - 1, int(xs.max()) + 2
    mask_crop = mask_bin[x0:x1, y0:y1]

    # Crop source and target into the same window
    ms0, ms1 = mask_on_src[0] + x0, mask_on_src[1] + y0
    src_crop = src[ms0:ms0 + (x1-x0), ms1:ms1 + (y1-y0)].astype(np.float32)

    # For target, we might go out of bounds if there's an offset. 
    # We'll pad the target image to ensure we can always take a full crop.
    mt0, mt1 = mask_on_tgt[0] + x0, mask_on_tgt[1] + y0
    
    # Calculate required padding
    pad_t = max(0, -mt0)
    pad_b = max(0, (mt0 + (x1-x0)) - tgt.shape[0])
    pad_l = max(0, -mt1)
    pad_r = max(0, (mt1 + (y1-y0)) - tgt.shape[1])
    
    if pad_t > 0 or pad_b > 0 or pad_l > 0 or pad_r > 0:
        tgt_padded = np.pad(tgt, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode='edge')
        mt0 += pad_t
        mt1 += pad_l
        tgt_crop = tgt_padded[mt0:mt0 + (x1-x0), mt1:mt1 + (y1-y0)].astype(np.float32)
    else:
        tgt_crop = tgt[mt0:mt0 + (x1-x0), mt1:mt1 + (y1-y0)].astype(np.float32)

    N, M = mask_crop.shape

    # Gradient of source (Poisson RHS)
    grad = np.zeros((N, M, 3), np.float32)
    grad[1:]  += src_crop[1:]  - src_crop[:-1]
    grad[:-1] += src_crop[:-1] - src_crop[1:]
    grad[:, 1:]  += src_crop[:, 1:]  - src_crop[:, :-1]
    grad[:, :-1] += src_crop[:, :-1] - src_crop[:, 1:]
    grad[mask_crop == 0] = 0

    # Bake Dirichlet boundary conditions into grad
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        m_shifted = np.roll(mask_crop, (-dy, -dx), axis=(0, 1))
        # zero the wrapped-around edge
        if dy == -1: m_shifted[-1, :] = 0
        if dy ==  1: m_shifted[0,  :] = 0
        if dx == -1: m_shifted[:, -1] = 0
        if dx ==  1: m_shifted[:,  0] = 0

        t_shifted = np.roll(tgt_crop, (-dy, -dx), axis=(0, 1))
        if dy == -1: t_shifted[-1, :, :] = 0
        if dy ==  1: t_shifted[0,  :, :] = 0
        if dx == -1: t_shifted[:, -1, :] = 0
        if dx ==  1: t_shifted[:,  0, :] = 0

        # pixels that are in mask but neighbour is NOT in mask
        boundary = (mask_crop == 1) & (m_shifted == 0)
        grad[boundary] += t_shifted[boundary]

    grad[mask_crop == 0] = 0

    return mask_crop, tgt_crop, grad, tgt.copy(), (mt0, mt0 + (x1-x0), mt1, mt1 + (y1-y0))

# ─────────────────────────────────────────────────────────────────────────────
# Benchmarking Solver
# ─────────────────────────────────────────────────────────────────────────────
def benchmark_opencl(mask_crop, tgt_crop, grad, n_iters, lwx, lwy):
    N, M = mask_crop.shape
    C = 3
    
    platforms = cl.get_platforms()
    ctx = None
    for plat in platforms:
        try:
            ctx = cl.Context(devices=plat.get_devices(cl.device_type.GPU), properties=[(cl.context_properties.PLATFORM, plat)])
            break
        except: pass
    if not ctx: ctx = cl.create_some_context(interactive=False)
    
    # Enable PROFILING
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    prog = cl.Program(ctx, KERNEL_SOURCE).build()
    
    kernels = {
        'init': cl.Kernel(prog, "kernel_init_r"),
        'Ap':   cl.Kernel(prog, "kernel_Ap"),
        'dot':  cl.Kernel(prog, "kernel_dot"),
        'uxr':  cl.Kernel(prog, "kernel_update_xr"),
        'up':   cl.Kernel(prog, "kernel_update_p")
    }

    mf = cl.mem_flags
    tgt_cm  = np.ascontiguousarray(tgt_crop.transpose(2, 0, 1), np.float32)
    grad_cm = np.ascontiguousarray(grad.transpose(2, 0, 1),     np.float32)
    mask_i  = np.ascontiguousarray(mask_crop,                   np.int32)

    # 1. Measure Host -> Device transfer
    t_h2d_start = time.time()
    buf_tgt  = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=tgt_cm)
    buf_grad = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=grad_cm)
    buf_mask = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=mask_i)
    buf_r    = cl.Buffer(ctx, mf.READ_WRITE, size=tgt_cm.nbytes)
    buf_p    = cl.Buffer(ctx, mf.READ_WRITE, size=tgt_cm.nbytes)
    buf_Ap   = cl.Buffer(ctx, mf.READ_WRITE, size=tgt_cm.nbytes)
    queue.finish()
    t_h2d = (time.time() - t_h2d_start) * 1000.0 # ms

    GWX = int(np.ceil(M / lwx)) * lwx
    GWY = int(np.ceil(N / lwy)) * lwy
    n_groups = (GWX // lwx) * (GWY // lwy)
    buf_dot = cl.Buffer(ctx, mf.READ_WRITE, size=n_groups * 4)
    partials = np.zeros(n_groups, np.float32)

    stats = {"kernel_ms": 0.0, "dot_ms": 0.0}

    def run_kernel(name, gsize, lsize, *args):
        ev = kernels[name](queue, gsize, lsize, *args)
        ev.wait()
        return (ev.profile.end - ev.profile.start) / 1e6 # ns -> ms

    def dot(buf_a, buf_b, c):
        ms = run_kernel('dot', (GWX, GWY), (lwx, lwy), buf_a, buf_b, buf_mask, buf_dot, cl.LocalMemory(lwx*lwy*4), np.int32(N), np.int32(M), np.int32(c))
        stats["dot_ms"] += ms
        cl.enqueue_copy(queue, partials, buf_dot).wait()
        return float(partials.sum())

    # Initialise
    stats["kernel_ms"] += run_kernel('init', (GWX, GWY, C), (lwx, lwy, 1), buf_tgt, buf_grad, buf_mask, buf_r, buf_p, np.int32(N), np.int32(M))
    r_sq = [dot(buf_r, buf_r, c) for c in range(C)]

    # Main Loop
    for it in range(n_iters):
        stats["kernel_ms"] += run_kernel('Ap', (GWX, GWY, C), (lwx, lwy, 1), buf_p, buf_mask, buf_Ap, np.int32(N), np.int32(M))
        pAp = [dot(buf_p, buf_Ap, c) for c in range(C)]
        alpha = [r_sq[c] / pAp[c] if pAp[c] > 1e-12 else 0.0 for c in range(C)]
        
        for c in range(C):
            stats["kernel_ms"] += run_kernel('uxr', (GWX, GWY), (lwx, lwy), buf_tgt, buf_r, buf_p, buf_Ap, buf_mask, np.float32(alpha[c]), np.int32(N), np.int32(M), np.int32(c))
        
        r_sq_new = [dot(buf_r, buf_r, c) for c in range(C)]
        beta = [r_sq_new[c] / r_sq[c] if r_sq[c] > 1e-12 else 0.0 for c in range(C)]
        r_sq = r_sq_new
        
        for c in range(C):
            stats["kernel_ms"] += run_kernel('up', (GWX, GWY), (lwx, lwy), buf_p, buf_r, buf_mask, np.float32(beta[c]), np.int32(N), np.int32(M), np.int32(c))

    # Device -> Host transfer
    t_d2h_start = time.time()
    result_cm = np.empty_like(tgt_cm)
    cl.enqueue_copy(queue, result_cm, buf_tgt).wait()
    t_d2h = (time.time() - t_d2h_start) * 1000.0 # ms

    return {
        "h2d_ms": t_h2d,
        "d2h_ms": t_d2h,
        "kernel_total_ms": stats["kernel_ms"] + stats["dot_ms"],
        "kernel_only_ms": stats["kernel_ms"],
        "dot_only_ms": stats["dot_ms"],
        "total_time_ms": (stats["kernel_ms"] + stats["dot_ms"] + t_h2d + t_d2h)
    }

def main():
    test_dir = "tests"
    configs = [(8, 8), (16, 16), (32, 8), (16, 32)]
    iters = 100
    
    print(f"{'Test':<10} | {'WG Size':<10} | {'H2D (ms)':<10} | {'D2H (ms)':<10} | {'Kernel (ms)':<12} | {'Total (ms)':<10}")
    print("-" * 75)
    
    tests = [
        {"name": "test0", "src_off": (0, 0), "tgt_off": (0, 0)},
        {"name": "test1", "src_off": (0, 0), "tgt_off": (0, 0)},
        {"name": "test2", "src_off": (0, 0), "tgt_off": (0, 0)},
        {"name": "test4", "src_off": (0, 0), "tgt_off": (0, 0)},
        {"name": "test5", "src_off": (0, 0), "tgt_off": (0, 0)},
    ]
    
    for tcfg in tests:
        tname = tcfg["name"]
        src_p = os.path.join(test_dir, f"{tname}_src.png")
        msk_p = os.path.join(test_dir, f"{tname}_mask.png")
        tgt_p = os.path.join(test_dir, f"{tname}_target.png")
        
        if not all(os.path.exists(p) for p in [src_p, msk_p, tgt_p]):
            continue
            
        src = cv2.imread(src_p)
        msk = cv2.imread(msk_p)
        tgt = cv2.imread(tgt_p)
        
        prep = preprocess(src, msk, tgt, tcfg["src_off"], tcfg["tgt_off"])
        if prep is None: continue
        mask_crop, tgt_crop, grad, _, _ = prep
        
        for lwx, lwy in configs:
            try:
                res = benchmark_opencl(mask_crop, tgt_crop, grad, iters, lwx, lwy)
                wg_str = f"{lwx}x{lwy}"
                print(f"{tname:<10} | {wg_str:<10} | {res['h2d_ms']:10.2f} | {res['d2h_ms']:10.2f} | {res['kernel_total_ms']:12.2f} | {res['total_time_ms']:10.2f}")
            except Exception as e:
                print(f"{tname:<10} | {lwx}x{lwy:<8} | ERROR: {str(e)[:30]}")

if __name__ == "__main__":
    main()

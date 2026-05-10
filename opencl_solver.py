"""
Standalone OpenCL Poisson Image Editing Solver
===============================================
No CMake, no C++, no PyBind11 required.

Install deps once:
    pip install pyopencl numpy opencv-python-headless

Run:
    python opencl_solver.py \
        --src test2_src.png \
        --mask test2_mask.png \
        --tgt test2_target.png \
        --out result_opencl.jpg \
        --src_offset 0 0 \
        --tgt_offset 260 260 \
        --iters 1000
"""

import argparse
import sys
import time

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# OpenCL kernels (verbatim copy of solver.cl with a corrected dot-product
# kernel and an added initialise-residual kernel)
# ─────────────────────────────────────────────────────────────────────────────
KERNEL_SOURCE = """
// =========================================================================
// KERNEL 0: Initialise residual and search direction
// r = b - Ax0  (b = grad + boundary;  A = Laplacian)
// p = r
// =========================================================================
__kernel void kernel_init_r(
    __global const float* tgt,   // current solution x (channel-major: [C][N][M])
    __global const float* grad,  // RHS gradient field  (same layout)
    __global const int*   mask,  // 0/1 mask (N*M ints)
    __global float*       r,     // residual output
    __global float*       p,     // search-direction output
    const int N, const int M)
{
    int px = get_global_id(0);   // column
    int py = get_global_id(1);   // row
    int c  = get_global_id(2);   // channel

    if (px >= M || py >= N) return;

    int id  = py * M + px;
    int gid = c * N * M + id;

    if (!mask[id]) {
        r[gid] = 0.0f;
        p[gid] = 0.0f;
        return;
    }

    // Laplacian Ax:  4*x_c - x_t - x_b - x_l - x_r
    // Boundary pixels outside the mask contribute their tgt value directly
    float x_c = tgt[gid];
    float x_t = (py > 0   && mask[id - M]) ? tgt[c*N*M + id - M] : 0.0f;
    float x_b = (py < N-1 && mask[id + M]) ? tgt[c*N*M + id + M] : 0.0f;
    float x_l = (px > 0   && mask[id - 1]) ? tgt[c*N*M + id - 1] : 0.0f;
    float x_r = (px < M-1 && mask[id + 1]) ? tgt[c*N*M + id + 1] : 0.0f;

    // Fixed Dirichlet contribution already baked into grad by the CPU
    float Ax = 4.0f * x_c - x_t - x_b - x_l - x_r;
    float res = grad[gid] - Ax;

    r[gid] = res;
    p[gid] = res;
}

// =========================================================================
// KERNEL 1: Ap = A * p   (Sparse Mat-Vec, Laplacian on search direction)
// =========================================================================
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

// =========================================================================
// KERNEL 2: Element-wise dot product reduction  (a . b, masked)
// Returns partial sums; CPU finishes the tiny final sum.
// =========================================================================
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

// =========================================================================
// KERNEL 3: Update solution x and residual r
//   x += alpha * p
//   r -= alpha * Ap
// =========================================================================
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

// =========================================================================
// KERNEL 4: Update search direction   p = r + beta * p
// =========================================================================
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
# Image I/O
# ─────────────────────────────────────────────────────────────────────────────
def read_images(src_path, mask_path, tgt_path):
    src  = cv2.imread(src_path)
    mask = cv2.imread(mask_path)
    tgt  = cv2.imread(tgt_path)
    if src is None or mask is None or tgt is None:
        raise FileNotFoundError("One or more image files not found.")
    # BGR -> RGB
    src  = cv2.cvtColor(src,  cv2.COLOR_BGR2RGB)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    tgt  = cv2.cvtColor(tgt,  cv2.COLOR_BGR2RGB)
    return src, mask, tgt


def write_image(path, img):
    cv2.imwrite(path, cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR))
    print(f"Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing (mirrors GridProcessor.reset in process.py)
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
    x0, x1 = int(ys.min()) - 1, int(ys.max()) + 2
    y0, y1 = int(xs.min()) - 1, int(xs.max()) + 2
    mask_crop = mask_bin[x0:x1, y0:y1]

    # Crop source and target into the same window
    ms0 = mask_on_src[0] + x0
    ms1 = mask_on_src[1] + y0
    src_crop = src[ms0:ms0 + (x1-x0), ms1:ms1 + (y1-y0)].astype(np.float32)

    mt0 = mask_on_tgt[0] + x0
    mt1 = mask_on_tgt[1] + y0
    tgt_crop = tgt[mt0:mt0 + (x1-x0), mt1:mt1 + (y1-y0)].astype(np.float32)

    N, M = mask_crop.shape

    # Gradient of source (Poisson RHS)
    grad = np.zeros((N, M, 3), np.float32)
    grad[1:]  += src_crop[1:]  - src_crop[:-1]
    grad[:-1] += src_crop[:-1] - src_crop[1:]
    grad[:, 1:]  += src_crop[:, 1:]  - src_crop[:, :-1]
    grad[:, :-1] += src_crop[:, :-1] - src_crop[:, 1:]
    grad[mask_crop == 0] = 0

    # Bake Dirichlet boundary conditions into grad:
    # For every masked pixel whose neighbour is *outside* the mask,
    # that neighbour's tgt value is a known constant → move to RHS.
    g = grad
    m = mask_crop
    t = tgt_crop
    # top neighbour
    bnd = (m == 1) & (np.pad(m, ((1, 0), (0, 0)), constant_values=0)[:-1] == 0)
    g[:, :, 0][bnd] += t[:, :, 0][np.pad(m, ((1,0),(0,0)), constant_values=0)[:-1] == 0] if False else 0
    # ... compact vectorised version:
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        m_shifted = np.roll(m, (-dy, -dx), axis=(0, 1))
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
        boundary = (m == 1) & (m_shifted == 0)
        g[boundary] += t_shifted[boundary]

    grad[mask_crop == 0] = 0

    return mask_crop, tgt_crop, grad, tgt.copy(), (mt0, mt0 + (x1-x0), mt1, mt1 + (y1-y0))


# ─────────────────────────────────────────────────────────────────────────────
# OpenCL CG Solver
# ─────────────────────────────────────────────────────────────────────────────
def run_opencl_cg(mask_crop, tgt_crop, grad, n_iters, chunk=100):
    try:
        import pyopencl as cl
    except ImportError:
        print("ERROR: pyopencl not installed.  Run:  pip install pyopencl")
        sys.exit(1)

    N, M = mask_crop.shape
    C = 3

    # ── Device setup ──────────────────────────────────────────────────────────
    platforms = cl.get_platforms()
    ctx = None
    for plat in platforms:
        try:
            ctx = cl.Context(
                devices=plat.get_devices(cl.device_type.GPU),
                properties=[(cl.context_properties.PLATFORM, plat)]
            )
            print(f"Using GPU: {plat.get_devices(cl.device_type.GPU)[0].name}")
            break
        except (cl.Error, IndexError):
            pass
    if ctx is None:
        ctx = cl.create_some_context(interactive=False)
        print(f"Using: {ctx.devices[0].name}")

    queue = cl.CommandQueue(ctx)

    # ── Compile kernels ───────────────────────────────────────────────────────
    prog = cl.Program(ctx, KERNEL_SOURCE).build()

    # Cache kernel handles (avoids RepeatedKernelRetrieval warnings)
    k_init   = cl.Kernel(prog, "kernel_init_r")
    k_Ap     = cl.Kernel(prog, "kernel_Ap")
    k_dot    = cl.Kernel(prog, "kernel_dot")
    k_upd_xr = cl.Kernel(prog, "kernel_update_xr")
    k_upd_p  = cl.Kernel(prog, "kernel_update_p")

    mf = cl.mem_flags

    # Channel-major layout: [C, N, M]
    # tgt_cm[c, y, x] = tgt_crop[y, x, c]
    tgt_cm  = np.ascontiguousarray(tgt_crop.transpose(2, 0, 1), np.float32)
    grad_cm = np.ascontiguousarray(grad.transpose(2, 0, 1),     np.float32)
    mask_i  = np.ascontiguousarray(mask_crop,                   np.int32)

    buf_tgt  = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=tgt_cm)
    buf_grad = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=grad_cm)
    buf_mask = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=mask_i)
    buf_r    = cl.Buffer(ctx, mf.READ_WRITE, size=tgt_cm.nbytes)
    buf_p    = cl.Buffer(ctx, mf.READ_WRITE, size=tgt_cm.nbytes)
    buf_Ap   = cl.Buffer(ctx, mf.READ_WRITE, size=tgt_cm.nbytes)

    # Dot-product partial sums
    LWX, LWY = 16, 16
    GWX = int(np.ceil(M / LWX)) * LWX
    GWY = int(np.ceil(N / LWY)) * LWY
    n_groups = (GWX // LWX) * (GWY // LWY)
    buf_dot = cl.Buffer(ctx, mf.READ_WRITE, size=n_groups * np.dtype(np.float32).itemsize)
    partials = np.zeros(n_groups, np.float32)

    local_size  = (LWX, LWY)
    global_size = (GWX, GWY)
    local_mem   = cl.LocalMemory(LWX * LWY * np.dtype(np.float32).itemsize)

    def dot(buf_a, buf_b, c):
        k_dot(
            queue, global_size, local_size,
            buf_a, buf_b, buf_mask, buf_dot, local_mem,
            np.int32(N), np.int32(M), np.int32(c)
        )
        cl.enqueue_copy(queue, partials, buf_dot)
        queue.finish()
        return float(partials.sum())

    # ── Initialise residuals r = b - Ax, p = r ────────────────────────────────
    k_init(
        queue, (GWX, GWY, C), (LWX, LWY, 1),
        buf_tgt, buf_grad, buf_mask, buf_r, buf_p,
        np.int32(N), np.int32(M)
    )
    queue.finish()

    # Initial r·r per channel
    r_sq = [dot(buf_r, buf_r, c) for c in range(C)]

    # ── Main CG loop ──────────────────────────────────────────────────────────
    print(f"Grid {N}×{M}, {sum(mask_crop.ravel())} active pixels, {n_iters} iters")
    t0 = time.time()

    for it in range(n_iters):
        # 1. Ap = A * p
        k_Ap(
            queue, (GWX, GWY, C), (LWX, LWY, 1),
            buf_p, buf_mask, buf_Ap,
            np.int32(N), np.int32(M)
        )
        queue.finish()

        # 2. alpha = r·r / p·Ap  (per channel)
        pAp = [dot(buf_p, buf_Ap, c) for c in range(C)]
        alpha = [r_sq[c] / pAp[c] if pAp[c] > 1e-12 else 0.0 for c in range(C)]

        # 3. x += alpha*p,  r -= alpha*Ap
        for c in range(C):
            k_upd_xr(
                queue, global_size, local_size,
                buf_tgt, buf_r, buf_p, buf_Ap, buf_mask,
                np.float32(alpha[c]),
                np.int32(N), np.int32(M), np.int32(c)
            )
        queue.finish()

        # 4. beta = r_new·r_new / r_old·r_old  (per channel)
        r_sq_new = [dot(buf_r, buf_r, c) for c in range(C)]
        beta = [r_sq_new[c] / r_sq[c] if r_sq[c] > 1e-12 else 0.0 for c in range(C)]
        r_sq = r_sq_new

        # 5. p = r + beta*p
        for c in range(C):
            k_upd_p(
                queue, global_size, local_size,
                buf_p, buf_r, buf_mask,
                np.float32(beta[c]),
                np.int32(N), np.int32(M), np.int32(c)
            )
        queue.finish()

        # Progress
        if (it + 1) % chunk == 0:
            err = [np.sqrt(r_sq[c]) for c in range(C)]
            elapsed = time.time() - t0
            print(f"  iter {it+1:4d} | ‖r‖ R={err[0]:.2f} G={err[1]:.2f} B={err[2]:.2f} | {elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.2f}s")

    # ── Read back solution ────────────────────────────────────────────────────
    result_cm = np.empty_like(tgt_cm)
    cl.enqueue_copy(queue, result_cm, buf_tgt)
    queue.finish()

    # [C, N, M] → [N, M, C]
    result = result_cm.transpose(1, 2, 0)
    return np.clip(result, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Standalone OpenCL Poisson Solver")
    parser.add_argument("--src",        default="test2_src.png")
    parser.add_argument("--mask",       default="test2_mask.png")
    parser.add_argument("--tgt",        default="test2_target.png")
    parser.add_argument("--out",        default="result_opencl.jpg")
    parser.add_argument("--src_offset", nargs=2, type=int, default=[0, 0],   metavar=("R", "C"))
    parser.add_argument("--tgt_offset", nargs=2, type=int, default=[260, 260], metavar=("R", "C"))
    parser.add_argument("--iters",      type=int, default=1000)
    parser.add_argument("--chunk",      type=int, default=100, help="Progress print frequency")
    args = parser.parse_args()

    print("Loading images…")
    src, mask_img, tgt = read_images(args.src, args.mask, args.tgt)

    print("Preprocessing…")
    mask_crop, tgt_crop, grad, tgt_full, (x0, x1, y0, y1) = preprocess(
        src, mask_img, tgt,
        tuple(args.src_offset),
        tuple(args.tgt_offset),
    )

    print("Running OpenCL CG solver…")
    result_crop = run_opencl_cg(mask_crop, tgt_crop, grad, args.iters, args.chunk)

    # Paste result back into the full target image
    tgt_full[x0:x1, y0:y1] = result_crop
    write_image(args.out, tgt_full)


if __name__ == "__main__":
    main()

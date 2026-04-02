#![allow(non_snake_case)]
//! Labrador sub-protocol implementation using ICICLE primitives.
//!
//! This is a translation of the lattirust-based Labrador implementation into
//! ICICLE's PolyRing / DeviceVec / matrix_ops / balanced_decomposition APIs.
//!
//! Reference: https://github.com/lattirust/labrador

use icicle_core::{
    matrix_ops::{self, MatMulConfig},
    vec_ops::VecOpsConfig,
    traits::GenerateRandom,
    balanced_decomposition,
    polynomial_ring::PolynomialRing,
    negacyclic_ntt,
    ntt::NTTDir,
    bignum::BigNum,
};
use icicle_babykoala::{
    ring::ScalarRing as Zq,
    polynomial_ring::PolyRing,
};
use icicle_runtime::{
    memory::{DeviceVec, HostSlice},
    IcicleError,
};
use spongefish::{
    protocol_id, DomainSeparator, ProverState, VerifierState,
};

// ============================================================================
// COMMON REFERENCE STRING
// ============================================================================

/// Common Reference String for one round of the Labrador protocol.
/// All matrices are stored flat in row-major order on the host.
#[derive(Clone)]
pub struct LabradorCRS {
    /// Number of witness vectors
    pub r: usize,
    /// Length of each witness vector (in PolyRing elements)
    pub n: usize,
    /// Ring degree (PolyRing::DEGREE)
    pub d: usize,
    /// Squared L2-norm bound on the concatenation of witness vectors
    pub norm_bound_sq: f64,
    /// Rank of inner commitment matrix A (k × n)
    pub k: usize,
    /// Rank of outer commitment matrix B (k1 × t1·r·k)
    pub k1: usize,
    /// Rank of outer commitment matrix D (k2 × t1·r(r+1)/2)
    pub k2: usize,
    /// Decomposition length in base b1
    pub t1: usize,
    /// Decomposition length in base b2
    pub t2: usize,
    /// Main decomposition basis (for z)
    pub b: u32,
    /// First auxiliary basis (for t and h decompositions)
    pub b1: u32,
    /// Second auxiliary basis (for g decomposition)
    pub b2: u32,
    /// Number of aggregated constraints (≈ ⌈128/log₂q⌉)
    pub num_aggregs: usize,
    /// Number of quadratic-linear constraints
    pub num_constraints: usize,

    /// Inner commitment matrix A: k × n, flat row-major
    pub A: Vec<PolyRing>,
    /// Outer commitment matrix B: k1 × (t1·r·k), flat row-major
    pub B: Vec<PolyRing>,
    /// Outer commitment matrix C: k2 × (t2·r(r+1)/2), flat row-major
    pub C: Vec<PolyRing>,
    /// Outer commitment matrix D: k1 × (t1·r(r+1)/2), flat row-major
    pub D_mat: Vec<PolyRing>,
}

/// Labrador witness: r vectors of length n each.
#[derive(Clone)]
pub struct LabradorWitness {
    pub s: Vec<Vec<PolyRing>>,
}

/// Labrador proof output from one round of the prover.
pub struct LabradorProof {
    pub u_1: Vec<PolyRing>,
    pub p: Vec<Zq>,
    pub b_agg: Vec<PolyRing>,
    pub u_2: Vec<PolyRing>,
    // TODO: removed z here, trying stuff
    //pub z: Vec<PolyRing>,
    pub t_flat: Vec<PolyRing>,
    pub G_flat: Vec<PolyRing>,
    pub H_flat: Vec<PolyRing>,
}

// ============================================================================
// SHARED UTILITIES
// ============================================================================

/// Compute inner products ⟨s_i, s_j⟩ for all i ≤ j (upper triangle).
/// Returns a flat array of r*(r+1)/2 PolyRing elements in row-major upper-triangular order.
pub fn inner_products_gpu(s_vecs: &[Vec<PolyRing>]) -> Result<Vec<PolyRing>, IcicleError> {
    let r = s_vecs.len();
    let n = s_vecs[0].len();
    let ntt_cfg = negacyclic_ntt::NegacyclicNttConfig::default();

    // To avoid OOM: process one row at a time.
    // For each s_i, compute its dot products with s_j (j >= i).
    
    // First, NTT-transform all s_i once and keep on host (or device if it fits).
    // Given the memory concern, let's keep them on host and move to device one by one.
    let mut s_ntt_host_vecs: Vec<Vec<PolyRing>> = Vec::with_capacity(r);
    for i in 0..r {
        let mut s_i_dev = DeviceVec::from_host_slice(&s_vecs[i]);
        negacyclic_ntt::ntt_inplace(&mut s_i_dev, NTTDir::kForward, &ntt_cfg)?;
        let mut s_i_ntt = vec![PolyRing::zero(); n];
        s_i_dev.copy_to_host(HostSlice::from_mut_slice(&mut s_i_ntt))?;
        s_ntt_host_vecs.push(s_i_ntt);
    }

    let mut result = Vec::with_capacity(r * (r + 1) / 2);
    for i in 0..r {
        let s_i_dev = DeviceVec::from_host_slice(&s_ntt_host_vecs[i]);
        
        // Compute dot products with s_j for j in [i, r-1]
        // We can batch these: s_i * S_j_tail^T
        let tail_len = r - i;
        let mut s_tail_flat = Vec::with_capacity(tail_len * n);
        for j in i..r {
            s_tail_flat.extend_from_slice(&s_ntt_host_vecs[j]);
        }
        
        let s_tail_dev = DeviceVec::from_host_slice(&s_tail_flat);
        let mut res_row_dev = DeviceVec::<PolyRing>::device_malloc(tail_len)?;
        let mut matcfg = MatMulConfig::default();
        matcfg.b_transposed = true; 
        
        matrix_ops::matmul::<PolyRing>(
            &s_i_dev, 1, n as u32,
            &s_tail_dev, tail_len as u32, n as u32,
            &matcfg, &mut res_row_dev,
        )?;
        
        negacyclic_ntt::ntt_inplace(&mut res_row_dev, NTTDir::kInverse, &ntt_cfg)?;
        let mut res_row_host = vec![PolyRing::zero(); tail_len];
        res_row_dev.copy_to_host(HostSlice::from_mut_slice(&mut res_row_host))?;
        result.extend_from_slice(&res_row_host);
    }

    Ok(result)
}

/// Matrix-vector multiplication on host: A (rows × cols) * v (cols × 1) → result (rows × 1).
/// Uses true polynomial multiplication mod (X^d + 1).
fn host_matmul(
    A: &[PolyRing], rows: usize, cols: usize,
    v: &[PolyRing],
) -> Result<Vec<PolyRing>, IcicleError> {
    let mut out_host = vec![PolyRing::zero(); rows];
    for row in 0..rows {
        let mut row_sum = PolyRing::zero();
        for col in 0..cols {
            let a_val = &A[row * cols + col];
            let v_val = &v[col];
            let prod = mul_poly_host(a_val, v_val);
            row_sum = add_poly_host(&row_sum, &prod);
        }
        out_host[row] = row_sum;
    }
    Ok(out_host)
}

/// Matrix-vector multiplication on device using NTT: A (rows × cols) * v (cols × 1) → result (rows × 1).
pub fn device_ntt_matmul(
    A_host: &[PolyRing], rows: usize, cols: usize,
    v_host: &[PolyRing],
) -> Result<Vec<PolyRing>, IcicleError> {
    let mut A_dev = DeviceVec::from_host_slice(A_host);
    let mut v_dev = DeviceVec::from_host_slice(v_host);

    let ntt_cfg = negacyclic_ntt::NegacyclicNttConfig::default();
    negacyclic_ntt::ntt_inplace(&mut A_dev, NTTDir::kForward, &ntt_cfg)?;
    negacyclic_ntt::ntt_inplace(&mut v_dev, NTTDir::kForward, &ntt_cfg)?;

    let mut out_dev = DeviceVec::<PolyRing>::device_malloc(rows)?;
    matrix_ops::matmul::<PolyRing>(
        &A_dev, rows as u32, cols as u32,
        &v_dev, cols as u32, 1,
        &MatMulConfig::default(),
        &mut out_dev,
    )?;

    negacyclic_ntt::ntt_inplace(&mut out_dev, NTTDir::kInverse, &ntt_cfg)?;

    let mut out_host = vec![PolyRing::zero(); rows];
    out_dev.copy_to_host(HostSlice::from_mut_slice(&mut out_host))?;
    Ok(out_host)
}

/// Helper for on-device matmul when inputs are already in NTT domain or on device.
pub fn device_ntt_matmul_dev(
    A_ntt: &DeviceVec<PolyRing>, rows: u32, cols: u32,
    v_ntt: &DeviceVec<PolyRing>,
) -> Result<DeviceVec<PolyRing>, IcicleError> {
    let mut out_dev = DeviceVec::<PolyRing>::device_malloc(rows as usize)?;
    matrix_ops::matmul::<PolyRing>(
        A_ntt, rows, cols,
        v_ntt, cols, 1,
        &MatMulConfig::default(),
        &mut out_dev,
    )?;
    Ok(out_dev)
}

/// Balanced decomposition on host: input → output (input_len * depth elements).
fn host_decompose(
    input: &[PolyRing],
    base: u32,
    depth: usize,
) -> Result<Vec<PolyRing>, IcicleError> {
    let input_dev = DeviceVec::from_host_slice(input);
    let out_len = input.len() * depth;
    let mut out_dev = DeviceVec::<PolyRing>::device_malloc(out_len)?;
    let cfg = VecOpsConfig::default();
    balanced_decomposition::decompose::<PolyRing>(
        &input_dev[..], &mut out_dev[..], base, &cfg,
    )?;
    let mut out_host = vec![PolyRing::zero(); out_len];
    out_dev.copy_to_host(HostSlice::from_mut_slice(&mut out_host))?;
    Ok(out_host)
}

/// Flatten a Vec<Vec<PolyRing>> into a single Vec<PolyRing>.
fn flatten_vecs(vecs: &[Vec<PolyRing>]) -> Vec<PolyRing> {
    vecs.iter().flat_map(|v| v.iter().cloned()).collect()
}

/// Elementwise addition of two host vectors.
fn vec_add_host(a: &[PolyRing], b: &[PolyRing]) -> Vec<PolyRing> {
    assert_eq!(a.len(), b.len());
    let d = PolyRing::DEGREE;
    let mut res = vec![PolyRing::zero(); a.len()];
    for i in 0..a.len() {
        let a_bytes = unsafe { std::slice::from_raw_parts(&a[i] as *const _ as *const Zq, d) };
        let b_bytes = unsafe { std::slice::from_raw_parts(&b[i] as *const _ as *const Zq, d) };
        let mut sum_coeffs = vec![Zq::zero(); d];
        for k in 0..d {
            sum_coeffs[k] = a_bytes[k] + b_bytes[k];
        }
        res[i] = PolyRing::from_slice(&sum_coeffs).unwrap();
    }
    res
}

/// Scalar multiplication of a host vector.
fn vec_scalar_mul_host(scalar: &PolyRing, v: &[PolyRing]) -> Vec<PolyRing> {
    let d = PolyRing::DEGREE;
    let mut res = vec![PolyRing::zero(); v.len()];
    let s_bytes = unsafe { std::slice::from_raw_parts(scalar as *const _ as *const Zq, d) };
    for i in 0..v.len() {
        let v_bytes = unsafe { std::slice::from_raw_parts(&v[i] as *const _ as *const Zq, d) };
        let mut prod = vec![Zq::zero(); d * 2];
        for a in 0..d {
            for b in 0..d {
                prod[a + b] = prod[a + b] + s_bytes[a] * v_bytes[b];
            }
        }
        let mut res_coeffs = vec![Zq::zero(); d];
        for a in 0..d {
            res_coeffs[a] = prod[a] - prod[a + d];
        }
        res[i] = PolyRing::from_slice(&res_coeffs).unwrap();
    }
    res
}

/// Dot product of two host vectors: Σ a[i] * b[i].
fn dot_product_host(a: &[PolyRing], b: &[PolyRing]) -> PolyRing {
    assert_eq!(a.len(), b.len());
    let d = PolyRing::DEGREE;
    let mut sum_coeffs = vec![Zq::zero(); d];
    for i in 0..a.len() {
        let a_bytes = unsafe { std::slice::from_raw_parts(&a[i] as *const _ as *const Zq, d) };
        let b_bytes = unsafe { std::slice::from_raw_parts(&b[i] as *const _ as *const Zq, d) };
        let mut prod = vec![Zq::zero(); d * 2];
        for k in 0..d {
            for l in 0..d {
                prod[k + l] = prod[k + l] + a_bytes[k] * b_bytes[l];
            }
        }
        for k in 0..d {
            sum_coeffs[k] = sum_coeffs[k] + prod[k] - prod[k + d];
        }
    }
    PolyRing::from_slice(&sum_coeffs).unwrap()
}

/// Single PolyRing multiplication on host (mod X^d + 1).
fn mul_poly_host(a: &PolyRing, b: &PolyRing) -> PolyRing {
    let d = PolyRing::DEGREE;
    let a_c = unsafe { std::slice::from_raw_parts(a as *const _ as *const Zq, d) };
    let b_c = unsafe { std::slice::from_raw_parts(b as *const _ as *const Zq, d) };
    let mut prod = vec![Zq::zero(); d * 2];
    for i in 0..d {
        for j in 0..d {
            prod[i + j] = prod[i + j] + a_c[i] * b_c[j];
        }
    }
    let mut res = vec![Zq::zero(); d];
    for i in 0..d { res[i] = prod[i] - prod[i + d]; }
    PolyRing::from_slice(&res).unwrap()
}

/// Multiply PolyRing by a scalar on host.
fn mul_poly_host_scalar(a: &PolyRing, s: Zq) -> PolyRing {
    let d = PolyRing::DEGREE;
    let a_c = unsafe { std::slice::from_raw_parts(a as *const _ as *const Zq, d) };
    let mut res = vec![Zq::zero(); d];
    for i in 0..d { res[i] = a_c[i] * s; }
    PolyRing::from_slice(&res).unwrap()
}

/// Single PolyRing addition on host.
fn add_poly_host(a: &PolyRing, b: &PolyRing) -> PolyRing {
    let d = PolyRing::DEGREE;
    let a_c = unsafe { std::slice::from_raw_parts(a as *const _ as *const Zq, d) };
    let b_c = unsafe { std::slice::from_raw_parts(b as *const _ as *const Zq, d) };
    let mut res = vec![Zq::zero(); d];
    for i in 0..d { res[i] = a_c[i] + b_c[i]; }
    PolyRing::from_slice(&res).unwrap()
}

/// Compare two PolyRing elements byte-for-byte.
fn poly_equal(a: &PolyRing, b: &PolyRing) -> bool {
    let size = std::mem::size_of::<PolyRing>();
    let a_bytes = unsafe { std::slice::from_raw_parts(a as *const _ as *const u8, size) };
    let b_bytes = unsafe { std::slice::from_raw_parts(b as *const _ as *const u8, size) };
    a_bytes == b_bytes
}

/// Compare two vectors of PolyRing elements.
fn poly_vecs_equal(a: &[PolyRing], b: &[PolyRing]) -> bool {
    if a.len() != b.len() { return false; }
    for i in 0..a.len() {
        if !poly_equal(&a[i], &b[i]) { return false; }
    }
    true
}

// ============================================================================
// CRS SETUP
// ============================================================================

impl LabradorCRS {
    /// Generate a new CRS with random matrices.
    pub fn setup(
        r: usize, n: usize,
        k: usize, k1: usize, k2: usize,
        t1: usize, t2: usize,
        b: u32, b1: u32, b2: u32,
        num_aggregs: usize,
        num_constraints: usize,
        norm_bound_sq: f64,
    ) -> Self {
        let d = PolyRing::DEGREE;

        // A: k × n
        let A = PolyRing::generate_random(k * n);
        // B: k1 × (t1 * r * k)
        let b_cols = t1 * r * k;
        let B = PolyRing::generate_random(k1 * b_cols);
        // C: k2 × (t2 * r*(r+1)/2)
        let c_cols = t2 * (r * (r + 1) / 2);
        let C = PolyRing::generate_random(k2 * c_cols);
        // D: k1 × (t1 * r*(r+1)/2)
        let d_cols = t1 * (r * (r + 1) / 2);
        let D_mat = PolyRing::generate_random(k1 * d_cols);

        LabradorCRS {
            r, n, d, norm_bound_sq,
            k, k1, k2, t1, t2,
            b, b1, b2,
            num_aggregs, num_constraints,
            A, B, C, D_mat,
        }
    }

    /// Width of B matrix.
    pub fn b_cols(&self) -> usize { self.t1 * self.r * self.k }
    /// Width of C matrix (symmetric inner products).
    pub fn c_cols(&self) -> usize { self.t2 * (self.r * (self.r + 1) / 2) }
    /// Width of D matrix.
    pub fn d_cols(&self) -> usize { self.t1 * (self.r * (self.r + 1) / 2) }
}

// ============================================================================
// PROVER — ONE ROUND
// ============================================================================

/// Labrador prover: one round of the protocol.
///
/// Takes the CRS, witness vectors s_1..s_r, and a ProverState for Fiat-Shamir.
/// Returns a LabradorProof.
pub fn labrador_prove_oneround(
    crs: &LabradorCRS,
    witness: &LabradorWitness,
    prover_state: &mut ProverState,
) -> Result<LabradorProof, IcicleError> {
    let cfg = VecOpsConfig::default();
    let r = crs.r;
    let n = crs.n;

    // =========================================================================
    // MESSAGE 1: Compute t_i = A·s_i, decompose, compute G = ⟨s_i, s_j⟩,
    //            decompose G, compute u_1 = B·t_flat + C·G_flat
    // =========================================================================

    // Compute all t_i = A * s_i on GPU in one batch: A(k x n) * S(n x r) = T(k x r)
    let mut s_flat = Vec::with_capacity(r * n);
    for s_i in &witness.s { s_flat.extend_from_slice(s_i); }
    
    let mut A_dev = DeviceVec::from_host_slice(&crs.A);
    let mut S_dev = DeviceVec::from_host_slice(&s_flat);
    let ntt_cfg = negacyclic_ntt::NegacyclicNttConfig::default();
    negacyclic_ntt::ntt_inplace(&mut A_dev, NTTDir::kForward, &ntt_cfg)?;
    negacyclic_ntt::ntt_inplace(&mut S_dev, NTTDir::kForward, &ntt_cfg)?;

    let mut T_dev = DeviceVec::<PolyRing>::device_malloc(crs.k * r)?;
    let mut matcfg_batch = MatMulConfig::default();
    matcfg_batch.b_transposed = true; // S is stored as r x n, we want A(k x n) * S^T(n x r)
    matrix_ops::matmul::<PolyRing>(&A_dev, crs.k as u32, n as u32, &S_dev, r as u32, n as u32, &matcfg_batch, &mut T_dev)?;
    negacyclic_ntt::ntt_inplace(&mut T_dev, NTTDir::kInverse, &ntt_cfg)?;

    // ---> HYBRID FIX: Reverted t_i formatting back to CPU so memory layout stays sequential <---
    let mut t_flat_host = vec![PolyRing::zero(); crs.k * r];
    T_dev.copy_to_host(HostSlice::from_mut_slice(&mut t_flat_host))?;
    
    let mut t_vecs: Vec<Vec<PolyRing>> = (0..r)
        .map(|i| {
            let mut v = vec![PolyRing::zero(); crs.k];
            for j in 0..crs.k { v[j] = t_flat_host[j * r + i]; }
            v
        })
        .collect();

    let mut t_decomp_flat: Vec<PolyRing> = Vec::with_capacity(crs.t1 * r * crs.k);
    for i in 0..r {
        let decomp_i = host_decompose(&t_vecs[i], crs.b1, crs.t1)?;
        t_decomp_flat.extend_from_slice(&decomp_i);
    }

    // =========================================================================
    // Compute G = ⟨s_i, s_j⟩ using a SINGLE batched GPU MatMul (S * S^T)
    // ---> KEEPS THE NEW ULTRA-FAST BATCHING FOR G! <---
    // =========================================================================
    let mut G_full_dev = DeviceVec::<PolyRing>::device_malloc(r * r)?;
    let mut matcfg_G = MatMulConfig::default();
    matcfg_G.b_transposed = true; // S(r x n) * S^T(n x r) = G(r x r)
    
    // S_dev is ALREADY in the NTT domain from the T_dev calculation above! Zero cost!
    matrix_ops::matmul::<PolyRing>(
        &S_dev, r as u32, n as u32, 
        &S_dev, r as u32, n as u32, 
        &matcfg_G, &mut G_full_dev
    )?;
    
    negacyclic_ntt::ntt_inplace(&mut G_full_dev, NTTDir::kInverse, &ntt_cfg)?;
    let mut G_full_host = vec![PolyRing::zero(); r * r];
    G_full_dev.copy_to_host(HostSlice::from_mut_slice(&mut G_full_host))?;

    // Extract the upper triangle of G into a flat array
    let mut G_flat_raw = Vec::with_capacity(r * (r + 1) / 2);
    for i in 0..r {
        for j in i..r {
            G_flat_raw.push(G_full_host[i * r + j]);
        }
    }

    // Decompose G
    let G_decomp = host_decompose(&G_flat_raw, crs.b2, crs.t2)?;

    // =========================================================================
    // Commit to t_flat and G_flat
    // =========================================================================
    let t_flat = t_decomp_flat.clone();
    let G_flat = G_decomp.clone();

    let u_1_part1 = device_ntt_matmul(&crs.B, crs.k1, crs.b_cols(), &t_flat)?;
    let u_1_part2 = device_ntt_matmul(&crs.C, crs.k2, crs.c_cols(), &G_flat)?;

    let mut u_1 = u_1_part1;
    for i in 0..u_1.len() {
        u_1[i] = add_poly_host(&u_1[i], &u_1_part2[i]);
    }

    // Absorb u_1 into Fiat-Shamir transcript
    let u_1_bytes: Vec<u8> = u_1.iter()
        .flat_map(|p| unsafe {
            std::slice::from_raw_parts(p as *const _ as *const u8, std::mem::size_of::<PolyRing>())
        }).cloned().collect();
    for &b in &u_1_bytes {
        prover_state.prover_message(&b);
    }

    // =========================================================================
    // CHALLENGE 1 & MESSAGE 2: Chunked JL projections
    // =========================================================================
    let num_projections = 256usize;
    let pi_seed = prover_state.verifier_message::<[u8; 32]>();
    let d = PolyRing::DEGREE;
    let cfg_zero = VecOpsConfig::default();

    println!("  [DEBUG] Starting chunked Labrador JL projections, R={}", r);
    
    // 1. PRE-ALLOCATE EVERYTHING OUTSIDE THE LOOP
    let pi_i_total = num_projections * n;
    let mut pi_i_dev = DeviceVec::<PolyRing>::device_malloc(pi_i_total)?;
    let mut p_i_dev = DeviceVec::<PolyRing>::device_malloc(num_projections)?;
    let mut s_i_dev = DeviceVec::<PolyRing>::device_malloc(n)?;
    
    // Accumulators
    let mut p_dev = DeviceVec::<PolyRing>::device_malloc(num_projections)?;
    let mut p_next = DeviceVec::<PolyRing>::device_malloc(num_projections)?;
    let p_zeros = vec![PolyRing::zero(); num_projections];
    p_dev.copy_from_host(HostSlice::from_slice(&p_zeros))?;

    for i in 0..r {
        if i % 10 == 0 { println!("  [DEBUG] JL projection witness {}", i); }
        
        let mut sub_seed = pi_seed;
        sub_seed[0] ^= (i & 0xFF) as u8;
        sub_seed[1] ^= ((i >> 8) & 0xFF) as u8;

        // 2. THE MAGIC: Sample Zq scalars directly into the PolyRing device memory
        {
            // This creates a safe "view" of the PolyRing memory as flat Zq scalars
            let mut pi_i_flat = icicle_core::polynomial_ring::flatten_polyring_slice_mut(&mut pi_i_dev);
            icicle_core::random_sampling::random_sampling(
                true, &sub_seed, &cfg, &mut pi_i_flat
            )?;
        } // The mutable borrow drops here, so pi_i_dev is safely a PolyRing vector again!

        // 3. Copy s_i directly into the pre-allocated buffer
        s_i_dev.copy_from_host(HostSlice::from_slice(&witness.s[i]))?;

        // 4. Transform to NTT domain in-place
        negacyclic_ntt::ntt_inplace(&mut pi_i_dev, NTTDir::kForward, &ntt_cfg)?;
        negacyclic_ntt::ntt_inplace(&mut s_i_dev, NTTDir::kForward, &ntt_cfg)?;

        // 5. Multiply
        matrix_ops::matmul::<PolyRing>(
            &pi_i_dev, num_projections as u32, n as u32,
            &s_i_dev, n as u32, 1,
            &MatMulConfig::default(),
            &mut p_i_dev,
        )?;
        
        // 6. Accumulate
        icicle_core::vec_ops::poly_vecops::polyvec_add::<PolyRing>(
            &p_dev, 
            &p_i_dev, 
            &mut p_next, 
            &cfg_zero
        )?;

        // 7. POINTER SWAP: No allocations, no copies. Just swap the labels!
        std::mem::swap(&mut p_dev, &mut p_next);
    }

    negacyclic_ntt::ntt_inplace(&mut p_dev, NTTDir::kInverse, &ntt_cfg)?;
    let mut p_poly_host = vec![PolyRing::zero(); num_projections];
    p_dev.copy_to_host(HostSlice::from_mut_slice(&mut p_poly_host))?;
    
    let p_host: Vec<Zq> = p_poly_host.iter().map(|p| {
        let coeffs = unsafe { std::slice::from_raw_parts(p as *const _ as *const Zq, d) };
        coeffs[0] // constant term
    }).collect();

    // Absorb p into transcript
    let p_bytes: Vec<u8> = p_host.iter()
        .flat_map(|s| unsafe {
            std::slice::from_raw_parts(s as *const _ as *const u8, std::mem::size_of::<Zq>())
        }).cloned().collect();
    for &b in &p_bytes {
        prover_state.prover_message(&b);
    }

    // =========================================================================
    // CHALLENGE 2: Squeeze psi and omega
    // =========================================================================
    let psi_omega_seed = prover_state.verifier_message::<[u8; 32]>();

    // =========================================================================
    // MESSAGE 3: Compute b'' (aggregated values)
    // =========================================================================
    // b''_k = Σ_{i,j} psi_k(i,j) * G(i,j)
    // Generate random psi coefficients from seed
    let g_entries = G_flat_raw.len(); // r*(r+1)/2
    let psi_total = crs.num_aggregs * g_entries * d;
    let mut psi_zq_dev = DeviceVec::<Zq>::device_malloc(psi_total)?;
    icicle_core::random_sampling::random_sampling(true, &psi_omega_seed, &cfg, &mut psi_zq_dev[..])?;
    let mut psi_zq_host = vec![Zq::zero(); psi_total];
    psi_zq_dev.copy_to_host(HostSlice::from_mut_slice(&mut psi_zq_host))?;
    let mut psi_host = vec![PolyRing::zero(); crs.num_aggregs * g_entries];
    for i in 0..psi_host.len() {
        psi_host[i] = PolyRing::from_slice(&psi_zq_host[i * d..(i + 1) * d]).unwrap();
    }

    // b_agg[k] = Σ_idx psi_{k,idx} * G(idx)
    // This is a matrix-vector product: psi * G_flat_raw
    let b_agg = device_ntt_matmul(&psi_host, crs.num_aggregs, g_entries, &G_flat_raw)?;

    // Absorb b'' into transcript
    let b_agg_bytes: Vec<u8> = b_agg.iter()
        .flat_map(|p| unsafe {
            std::slice::from_raw_parts(p as *const _ as *const u8, std::mem::size_of::<PolyRing>())
        }).cloned().collect();
    for &b in &b_agg_bytes {
        prover_state.prover_message(&b);
    }

    // =========================================================================
    // CHALLENGE 3: Squeeze alpha and beta
    // =========================================================================
    let alpha_beta_seed = prover_state.verifier_message::<[u8; 32]>();

    // =========================================================================
    // MESSAGE 4: Compute H (inner products with phi), decompose, u_2 = D·H_flat
    // =========================================================================
    // Generate random phi vectors from alpha_beta_seed: r vectors of length n
    let phi_total = r * n * d;
    let mut phi_zq_dev = DeviceVec::<Zq>::device_malloc(phi_total)?;
    icicle_core::random_sampling::random_sampling(true, &alpha_beta_seed, &cfg, &mut phi_zq_dev[..])?;
    let mut phi_zq_host = vec![Zq::zero(); phi_total];
    phi_zq_dev.copy_to_host(HostSlice::from_mut_slice(&mut phi_zq_host))?;
    let mut phi: Vec<Vec<PolyRing>> = Vec::with_capacity(r);
    for i in 0..r {
        let mut phi_i = vec![PolyRing::zero(); n];
        for j in 0..n {
            let offset = (i * n + j) * d;
            phi_i[j] = PolyRing::from_slice(&phi_zq_host[offset..offset + d]).unwrap();
        }
        phi.push(phi_i);
    }

    // H(i,j) = ⟨phi_i, s_j⟩ + ⟨phi_j, s_i⟩
    // This is the upper triangle of (phi * S^T) + (S * phi^T)
    let mut phi_flat = Vec::with_capacity(r * n);
    for p_i in &phi { phi_flat.extend_from_slice(p_i); }
    let mut s_flat = Vec::with_capacity(r * n);
    for s_i in &witness.s { s_flat.extend_from_slice(s_i); }

    let mut phi_dev = DeviceVec::from_host_slice(&phi_flat);
    let mut s_dev = DeviceVec::from_host_slice(&s_flat);
    let ntt_cfg = negacyclic_ntt::NegacyclicNttConfig::default();
    negacyclic_ntt::ntt_inplace(&mut phi_dev, NTTDir::kForward, &ntt_cfg)?;
    negacyclic_ntt::ntt_inplace(&mut s_dev, NTTDir::kForward, &ntt_cfg)?;

    let mut mat_res_dev = DeviceVec::<PolyRing>::device_malloc(r * r)?;
    let mut matcfg = MatMulConfig::default();
    matcfg.b_transposed = true; // Phi * S^T
    matrix_ops::matmul::<PolyRing>(&phi_dev, r as u32, n as u32, &s_dev, r as u32, n as u32, &matcfg, &mut mat_res_dev)?;
    
    let mut mat_res_host = vec![PolyRing::zero(); r * r];
    negacyclic_ntt::ntt_inplace(&mut mat_res_dev, NTTDir::kInverse, &ntt_cfg)?;
    mat_res_dev.copy_to_host(HostSlice::from_mut_slice(&mut mat_res_host))?;

    let h_entries = r * (r + 1) / 2;
    let mut H_flat_raw = vec![PolyRing::zero(); h_entries];
    let mut h_idx = 0;
    for i in 0..r {
        for j in 0..=i {
            // H(i,j) = (Phi S^T)_{i,j} + (Phi S^T)_{j,i}
            H_flat_raw[h_idx] = add_poly_host(&mat_res_host[i * r + j], &mat_res_host[j * r + i]);
            h_idx += 1;
        }
    }
    let H_decomp = host_decompose(&H_flat_raw, crs.b1, crs.t1)?;
    let H_flat = H_decomp;

    // u_2 = D * H_flat
    let u_2 = device_ntt_matmul(&crs.D_mat, crs.k1, crs.d_cols(), &H_flat)?;

    // Absorb u_2
    let u_2_bytes: Vec<u8> = u_2.iter()
        .flat_map(|p| unsafe {
            std::slice::from_raw_parts(p as *const _ as *const u8, std::mem::size_of::<PolyRing>())
        }).cloned().collect();
    for &b in &u_2_bytes {
        prover_state.prover_message(&b);
    }

    Ok(LabradorProof {
        u_1,
        p: p_host,
        b_agg,
        u_2,
        t_flat,
        G_flat,
        H_flat,
    })
}

// ============================================================================
// VERIFIER — ONE ROUND
// ============================================================================

/// Labrador verifier: one round of the protocol.
///
/// Takes the CRS, proof, and VerifierState (initialized with the prover's narg_string).
/// Returns Ok(true) if verification passes.
pub fn labrador_verify_oneround(
    crs: &LabradorCRS,
    proof: &LabradorProof,
    verifier_state: &mut VerifierState,
    greyhound_c: &[PolyRing], // NEW
    greyhound_z: &[PolyRing], // NEW
) -> Result<bool, IcicleError> {
    let cfg = VecOpsConfig::default();
    let r = crs.r;
    let n = crs.n;

    // =========================================================================
    // STEP 1: Extract u_1 from transcript, reproduce Challenge 1
    // =========================================================================
    let u_1_bytes: Vec<u8> = proof.u_1.iter()
        .flat_map(|p| unsafe {
            std::slice::from_raw_parts(p as *const _ as *const u8, std::mem::size_of::<PolyRing>())
        }).cloned().collect();
    for _ in 0..u_1_bytes.len() {
        verifier_state.prover_message::<[u8; 1]>().unwrap();
    }

    // Reproduce JL seed
    let _pi_seed = verifier_state.verifier_message::<[u8; 32]>();

    // =========================================================================
    // STEP 2: Extract p from transcript, check norm bound
    // =========================================================================
    let p_bytes: Vec<u8> = proof.p.iter()
        .flat_map(|s| unsafe {
            std::slice::from_raw_parts(s as *const _ as *const u8, std::mem::size_of::<Zq>())
        }).cloned().collect();
    for _ in 0..p_bytes.len() {
        verifier_state.prover_message::<[u8; 1]>().unwrap();
    }

    // Reproduce psi/omega seed
    let _psi_omega_seed = verifier_state.verifier_message::<[u8; 32]>();

    // =========================================================================
    // STEP 3: Extract b'' from transcript
    // =========================================================================
    let b_agg_bytes: Vec<u8> = proof.b_agg.iter()
        .flat_map(|p| unsafe {
            std::slice::from_raw_parts(p as *const _ as *const u8, std::mem::size_of::<PolyRing>())
        }).cloned().collect();
    for _ in 0..b_agg_bytes.len() {
        verifier_state.prover_message::<[u8; 1]>().unwrap();
    }

    // Reproduce alpha/beta seed
    let _alpha_beta_seed = verifier_state.verifier_message::<[u8; 32]>();

    // =========================================================================
    // STEP 4: Extract u_2, reproduce final challenge c
    // =========================================================================
    let u_2_bytes: Vec<u8> = proof.u_2.iter()
        .flat_map(|p| unsafe {
            std::slice::from_raw_parts(p as *const _ as *const u8, std::mem::size_of::<PolyRing>())
        }).cloned().collect();
    for _ in 0..u_2_bytes.len() {
        verifier_state.prover_message::<[u8; 1]>().unwrap();
    }

    let c_seed = verifier_state.verifier_message::<[u8; 32]>();

    let d = PolyRing::DEGREE;
    let mut c_zq_dev = DeviceVec::<Zq>::device_malloc(r * d).unwrap();
    icicle_core::random_sampling::random_sampling(true, &c_seed, &cfg, &mut c_zq_dev[..]).unwrap();
    let mut c_zq_host = vec![Zq::zero(); r * d];
    c_zq_dev.copy_to_host(HostSlice::from_mut_slice(&mut c_zq_host)).unwrap();
    let mut c_host = vec![PolyRing::zero(); r];
    for i in 0..r {
        let chunk = &c_zq_host[i * d..(i + 1) * d];
        c_host[i] = PolyRing::from_slice(chunk).unwrap();
    }

    // =========================================================================
    // VERIFICATION CHECKS
    // =========================================================================

    // Check 1: u_1 == B * t_flat + C * G_flat
    let u_1_check_part1 = device_ntt_matmul(&crs.B, crs.k1, crs.b_cols(), &proof.t_flat)?;
    let u_1_check_part2 = device_ntt_matmul(&crs.C, crs.k2, crs.c_cols(), &proof.G_flat)?;
    let u_1_check = vec_add_host(&u_1_check_part1, &u_1_check_part2);
    if !poly_vecs_equal(&u_1_check, &proof.u_1) {
        println!("    [LAB FAIL] Check 1: u_1 != B*t_flat + C*G_flat");
        return Ok(false);
    }
    println!("    [LAB PASS] Check 1: u_1 == B*t_flat + C*G_flat");

    // Check 2: u_2 == D * H_flat
    let u_2_check = device_ntt_matmul(&crs.D_mat, crs.k1, crs.d_cols(), &proof.H_flat)?;
    if !poly_vecs_equal(&u_2_check, &proof.u_2) {
        println!("    [LAB FAIL] Check 2: u_2 != D*H_flat");
        return Ok(false);
    }
    println!("    [LAB PASS] Check 2: u_2 == D*H_flat");

    // Check 3: A * z == Σ c_i * (Σ hat_t_{i,digit} * b1^digit)
    let Az = device_ntt_matmul(&crs.A, crs.k, n, greyhound_z)?;
    
    // Compute Σ c_i * Σ hat_t_{i,digit} * b1^digit on GPU
    // Weights w_{i,digit} = c_i(X) * b1^digit
    let mut weights_host = vec![PolyRing::zero(); r * crs.t1];
    for i in 0..r {
        let mut b1_pow = Zq::one();
        for d_idx in 0..crs.t1 {
            let c_i_poly = greyhound_c[i];
            let c_i_coeffs = unsafe { std::slice::from_raw_parts(&c_i_poly as *const _ as *const Zq, d) };
            let mut w_coeffs = Vec::with_capacity(d);
            for k in 0..d { w_coeffs.push(c_i_coeffs[k] * b1_pow); }
            weights_host[i * crs.t1 + d_idx] = PolyRing::from_slice(&w_coeffs).unwrap();
            b1_pow = b1_pow * Zq::from(crs.b1);
        }
    }
    let mut weights_dev = DeviceVec::from_host_slice(&weights_host);
    let mut t_flat_dev = DeviceVec::from_host_slice(&proof.t_flat);
    let ntt_cfg = negacyclic_ntt::NegacyclicNttConfig::default();
    negacyclic_ntt::ntt_inplace(&mut weights_dev, NTTDir::kForward, &ntt_cfg)?;
    negacyclic_ntt::ntt_inplace(&mut t_flat_dev, NTTDir::kForward, &ntt_cfg)?;
    
    // t_flat is (R*T1) x K in row-major
    let mut sum_ct_dev = DeviceVec::<PolyRing>::device_malloc(crs.k)?;
    matrix_ops::matmul::<PolyRing>(
        &weights_dev, 1, (r * crs.t1) as u32,
        &t_flat_dev, (r * crs.t1) as u32, crs.k as u32,
        &MatMulConfig::default(),
        &mut sum_ct_dev,
    )?;
    
    negacyclic_ntt::ntt_inplace(&mut sum_ct_dev, NTTDir::kInverse, &ntt_cfg)?;
    let mut sum_ct = vec![PolyRing::zero(); crs.k];
    sum_ct_dev.copy_to_host(HostSlice::from_mut_slice(&mut sum_ct))?;
    if !poly_vecs_equal(&Az, &sum_ct) {
        println!("    [LAB FAIL] Check 3: A*z != Σ c_i*t_i");
        let az_bytes = unsafe { std::slice::from_raw_parts(&Az[0] as *const _ as *const u64, d) };
        let sum_ct_bytes = unsafe { std::slice::from_raw_parts(&sum_ct[0] as *const _ as *const u64, d) };
        println!("      Az[0] = 0x{:x}, sum_ct[0] = 0x{:x}", az_bytes[0], sum_ct_bytes[0]);
        return Ok(false);
    }
    println!("    [LAB PASS] Check 3: A*z == Σ c_i*t_i");

    // Check 4: ⟨z, z⟩ == Σ c_i c_j G(i,j)
    // Recompose G natively on the GPU
    let g_entries = r * (r + 1) / 2;
    let mut g_flat_dev = DeviceVec::from_host_slice(&proof.G_flat);
    let mut G_recomposed_dev = DeviceVec::<PolyRing>::device_malloc(g_entries)?;
    
    // Use ICICLE's native recompose to perfectly invert the decomposition
    icicle_core::balanced_decomposition::recompose::<PolyRing>(
        &g_flat_dev[..],
        &mut G_recomposed_dev[..],
        crs.b2,
        &VecOpsConfig::default()
    )?;
    
    let mut G_recomposed = vec![PolyRing::zero(); g_entries];
    G_recomposed_dev.copy_to_host(HostSlice::from_mut_slice(&mut G_recomposed))?;

    let z_dot_z = dot_product_host(greyhound_z, greyhound_z);
    
    let mut sum_cc_g = PolyRing::zero();
    let mut g_idx = 0;
    for i in 0..r {
        for j in i..r {
            // USE GREYHOUND'S C!
            let ci_cj = mul_poly_host(&greyhound_c[i], &greyhound_c[j]);
            let mut term = mul_poly_host(&ci_cj, &G_recomposed[g_idx]);
            if i != j {
                term = add_poly_host(&term, &term);
            }
            sum_cc_g = add_poly_host(&sum_cc_g, &term);
            g_idx += 1;
        }
    }
    if !poly_equal(&z_dot_z, &sum_cc_g) {
        println!("    [LAB FAIL] Check 4: <z,z> != Σ c_i c_j G(i,j)");
        let z_bytes = unsafe { std::slice::from_raw_parts(&z_dot_z as *const _ as *const u64, d) };
        let sum_bytes = unsafe { std::slice::from_raw_parts(&sum_cc_g as *const _ as *const u64, d) };
        println!("      <z,z> = 0x{:x}, Σ c_i c_j G = 0x{:x}", z_bytes[0], sum_bytes[0]);
        return Ok(false);
    }
    println!("    [LAB PASS] Check 4: <z,z> == Σ c_i c_j G(i,j)");

    Ok(true)
}

// ============================================================================
// MULTI-ROUND WRAPPER
// ============================================================================

/// Full Labrador proof protocol ID for Fiat-Shamir.
pub fn labrador_protocol_id() -> [u8; 64] {
    protocol_id(core::format_args!("labrador proof"))
}

/// Full Labrador prover: runs one round of the protocol (can be extended to recursive).
pub fn labrador_prove(
    crs: &LabradorCRS,
    witness: &LabradorWitness,
) -> Result<(LabradorProof, Vec<u8>), IcicleError> {
    let domain_sep = DomainSeparator::new(labrador_protocol_id())
        .session(spongefish::session!("labrador"))
        .instance(&[0u8; 0]);
    let mut prover_state = domain_sep.std_prover();

    let proof = labrador_prove_oneround(crs, witness, &mut prover_state)?;
    let transcript = prover_state.narg_string().to_vec();

    Ok((proof, transcript))
}

/// Full Labrador verifier: verifies one round of the protocol.
/// Full Labrador verifier: verifies one round of the protocol.
pub fn labrador_verify(
    crs: &LabradorCRS,
    proof: &LabradorProof,
    transcript: &[u8],
    greyhound_c: &[PolyRing], // NEW
    greyhound_z: &[PolyRing], // NEW
) -> Result<bool, IcicleError> {
    let domain_sep = DomainSeparator::new(labrador_protocol_id())
        .session(spongefish::session!("labrador"))
        .instance(&[0u8; 0]);
    let mut verifier_state = domain_sep.std_verifier(transcript);

    // Hand the variables down to the inner function
    match labrador_verify_oneround(crs, proof, &mut verifier_state, greyhound_c, greyhound_z) {
        Ok(res) => Ok(res),
        Err(_) => Ok(false),
    }
}

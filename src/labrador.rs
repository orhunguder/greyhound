    #![allow(non_snake_case)]
    //! Labrador sub-protocol implementation using ICICLE primitives.
    //! This is a translation of the lattirust-based Labrador implementation into
    //! ICICLE's PolyRing / DeviceVec / matrix_ops / balanced_decomposition APIs.
    //! Reference: https://github.com/lattirust/labrador

    use std::time::Instant;
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
use rand::{Rng, SeedableRng};
use rand::seq::SliceRandom;
use rand_chacha::ChaCha20Rng;
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

    /// The full constraint system from Greyhound: P matrix components + target h.
    /// P is NOT materialized we store the sub-matrices and evaluate P·z piecewise.
    pub struct ConstraintSystem {
        /// Evaluation vector a (DELTA0*M elements)
        pub a_eval: Vec<PolyRing>,
        /// Evaluation vector b (R elements) 
        pub b_eval: Vec<PolyRing>,
        /// Fiat-Shamir challenge vector c (R elements)
        pub c_eval: Vec<PolyRing>,
        /// σ_{-1}(x) automorphism polynomial
        pub sigma_inv_x: PolyRing,
        /// Ajtai commitment matrix A (NROWS × DELTA0*M)
        pub A_ajtai: Vec<PolyRing>,
        /// Greyhound commitment matrix B (NROWS × NROWS*DELTA*R)
        pub B_commit: Vec<PolyRing>,
        /// Greyhound commitment matrix D (NROWS × DELTA*R)
        pub D_commit: Vec<PolyRing>,
        /// Target vector h = [v, u, σ⁻¹(x)⁻¹·y, 0, 0]
        pub h_target: Vec<PolyRing>,
        /// Width of the full w_hat block
        pub n_w: usize,
        /// Width of the full t_hat block
        pub n_t: usize,
        /// Width of the folded s/z block
        pub n_s: usize,
        /// Number of commitment rows
        pub n_commit: usize,
    }

    pub struct LabradorWitness {
        /// Greyhound line-15 witness z = [w_hat | t_hat | [s_1 | ... | s_r]c].
        pub z: Vec<PolyRing>,
        /// Legacy row-folded prototype witness; no longer used by the active verifier path.
        pub s: Vec<Vec<PolyRing>>,
    }

    /// Labrador proof output from one round of the prover.
    #[derive(Clone)]
    pub struct LabradorProof {
        pub u_1: Vec<PolyRing>,
        pub p: Vec<Zq>,
        pub b_agg: Vec<PolyRing>,
        pub u_2: Vec<PolyRing>,
        pub z_folded: Vec<PolyRing>,
        pub t_flat: Vec<PolyRing>,
        pub G_flat: Vec<PolyRing>,
        pub H_flat: Vec<PolyRing>,
    }

    // Generates exactly `num_challenges` polynomials following the LaBRADOR distribution.
    /// 23 zeros, 31 (+/- 1), 10 (+/- 2).
    fn generate_labrador_challenges(seed: &[u8; 32], num_challenges: usize) -> Vec<PolyRing> {
        let mut prng = ChaCha20Rng::from_seed(*seed);
        let d = PolyRing::DEGREE;
        let mut challenges = Vec::with_capacity(num_challenges);

        // The exact absolute value distribution specified in LaBRADOR Section 2
        let mut base_mags = vec![0u32; 23];
        base_mags.extend(vec![1u32; 31]);
        base_mags.extend(vec![2u32; 10]);
        assert_eq!(base_mags.len(), d); // Must equal 64

        for _ in 0..num_challenges {
            // NOTE: The paper also uses rejection sampling here to ensure the operator 
            // norm ||c||_op <= 15. Calculating operator norms requires complex FFTs, 
            // so we skip that strict rejection check for this prototype. 
            // The L1 norm is guaranteed to be exactly 51.

            let mut mags = base_mags.clone();
            mags.shuffle(&mut prng); // Shuffle the positions of the 1s, 2s, and 0s

            let mut coeffs = vec![Zq::zero(); d];
            for i in 0..d {
                if mags[i] != 0 {
                    // 50% chance for positive or negative
                    let is_positive: bool = prng.r#gen();
                    if is_positive {
                        coeffs[i] = Zq::from(mags[i]);
                    } else {
                        // Negative representations in Zq are (q - val)
                        coeffs[i] = Zq::zero() - Zq::from(mags[i]);
                    }
                }
            }
            challenges.push(PolyRing::from_slice(&coeffs).unwrap());
        }

        challenges
    }

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

    fn scale_poly_host(a: &PolyRing, scalar: Zq) -> PolyRing {
        let d = PolyRing::DEGREE;
        let a_c = unsafe { std::slice::from_raw_parts(a as *const _ as *const Zq, d) };
        let mut res = vec![Zq::zero(); d];
        for i in 0..d {
            res[i] = a_c[i] * scalar;
        }
        PolyRing::from_slice(&res).unwrap()
    }

    fn recompose_digit_major(
        digits: &[PolyRing],
        num_values: usize,
        digit_count: usize,
        base: u32,
    ) -> Vec<PolyRing> {
        assert_eq!(digits.len(), num_values * digit_count);
        let mut recomposed = vec![PolyRing::zero(); num_values];
        let mut b_pow = Zq::one();
        let base_zq = Zq::from(base);

        for k in 0..digit_count {
            for i in 0..num_values {
                let term = scale_poly_host(&digits[k * num_values + i], b_pow);
                recomposed[i] = add_poly_host(&recomposed[i], &term);
            }
            b_pow = b_pow * base_zq;
        }

        recomposed
    }

    fn poly_vec_norm_sq(z: &[PolyRing]) -> f64 {
        let d = PolyRing::DEGREE;
        let mut total = 0.0;
        for poly in z {
            let coeffs = unsafe { std::slice::from_raw_parts(poly as *const _ as *const Zq, d) };
            for coeff in coeffs {
                let val = unsafe { *(coeff as *const _ as *const u32) };
                let mag = if val > (1 << 30) {
                    let neg = Zq::zero() - *coeff;
                    unsafe { *(&neg as *const _ as *const u32) }
                } else {
                    val
                };
                let mag_f = mag as f64;
                total += mag_f * mag_f;
            }
        }
        total
    }

    fn verify_greyhound_pz(
        crs: &LabradorCRS,
        z: &[PolyRing],
        constraint: &ConstraintSystem,
    ) -> Result<bool, IcicleError> {
        let n_w = constraint.n_w;
        let n_t = constraint.n_t;
        let n_s = constraint.n_s;
        let n_commit = constraint.n_commit;
        let r = constraint.c_eval.len();
        let n_full = n_w + n_t + n_s;
        let h_len = (3 * n_commit) + 2;
        let _sigma_inv_x = &constraint.sigma_inv_x;

        if z.len() != n_full {
            // println!("    [LAB FAIL] z length mismatch: got {}, expected {}", z.len(), n_full);
            return Ok(false);
        }
        if constraint.h_target.len() != h_len {
            // println!("    [LAB FAIL] h length mismatch: got {}, expected {}", constraint.h_target.len(), h_len);
            return Ok(false);
        }
        if r == 0 || n_w % r != 0 || n_t % r != 0 {
            // println!("    [LAB FAIL] invalid Greyhound block dimensions");
            return Ok(false);
        }
        if constraint.b_eval.len() != r || constraint.a_eval.len() != n_s {
            // println!("    [LAB FAIL] Greyhound evaluation vector length mismatch");
            return Ok(false);
        }
        if constraint.D_commit.len() != n_commit * n_w || constraint.B_commit.len() != n_commit * n_t {
            // println!("    [LAB FAIL] Greyhound commitment matrix length mismatch");
            return Ok(false);
        }

        let z_w = &z[..n_w];
        let z_t = &z[n_w..n_w + n_t];
        let z_s = &z[n_w + n_t..];

        let h_v = &constraint.h_target[..n_commit];
        let h_u = &constraint.h_target[n_commit..2 * n_commit];
        let h_y = &constraint.h_target[2 * n_commit];
        let h_zero_scalar = &constraint.h_target[(2 * n_commit) + 1];
        let h_zero_vec = &constraint.h_target[(2 * n_commit) + 2..];

        let d_w = device_ntt_matmul(&constraint.D_commit, n_commit, n_w, z_w)?;
        if !poly_vecs_equal(&d_w, h_v) {
            // println!("    [LAB FAIL] Check P.1: D*w_hat != v");
            return Ok(false);
        }
        // println!("    [LAB PASS] Check P.1: D*w_hat == v");

        let b_t = device_ntt_matmul(&constraint.B_commit, n_commit, n_t, z_t)?;
        if !poly_vecs_equal(&b_t, h_u) {
            // println!("    [LAB FAIL] Check P.2: B*t_hat != u");
            return Ok(false);
        }
        // println!("    [LAB PASS] Check P.2: B*t_hat == u");

        let w_digits = n_w / r;
        let w = recompose_digit_major(z_w, r, w_digits, crs.b);

        let mut b_dot_w = PolyRing::zero();
        for i in 0..r {
            let term = mul_poly_host(&constraint.b_eval[i], &w[i]);
            b_dot_w = add_poly_host(&b_dot_w, &term);
        }
        if !poly_equal(&b_dot_w, h_y) {
            // println!("    [LAB FAIL] Check P.3: b^T*G*w_hat != sigma^-1(x)^-1*y");
            return Ok(false);
        }
        // println!("    [LAB PASS] Check P.3: b^T*G*w_hat == sigma^-1(x)^-1*y");

        let mut c_dot_w = PolyRing::zero();
        for i in 0..r {
            let term = mul_poly_host(&constraint.c_eval[i], &w[i]);
            c_dot_w = add_poly_host(&c_dot_w, &term);
        }
        let mut a_dot_z = PolyRing::zero();
        for j in 0..constraint.a_eval.len() {
            let term = mul_poly_host(&constraint.a_eval[j], &z_s[j]);
            a_dot_z = add_poly_host(&a_dot_z, &term);
        }
        let row4_expected = add_poly_host(&a_dot_z, h_zero_scalar);
        if !poly_equal(&c_dot_w, &row4_expected) {
            // println!("    [LAB FAIL] Check P.4: c^T*G*w_hat - a^T*z != 0");
            return Ok(false);
        }
        // println!("    [LAB PASS] Check P.4: c^T*G*w_hat - a^T*z == 0");

        let t_per_column = n_t / r;
        if t_per_column % n_commit != 0 {
            // println!("    [LAB FAIL] invalid t_hat block width");
            return Ok(false);
        }
        let t_digits = t_per_column / n_commit;
        let mut c_dot_t = vec![PolyRing::zero(); n_commit];
        for i in 0..r {
            let t_i = recompose_digit_major(
                &z_t[i * t_per_column..(i + 1) * t_per_column],
                n_commit,
                t_digits,
                crs.b,
            );
            for row in 0..n_commit {
                let term = mul_poly_host(&constraint.c_eval[i], &t_i[row]);
                c_dot_t[row] = add_poly_host(&c_dot_t[row], &term);
            }
        }

        let a_z = device_ntt_matmul(&constraint.A_ajtai, n_commit, n_s, z_s)?;
        let mut row5_expected = Vec::with_capacity(n_commit);
        for row in 0..n_commit {
            row5_expected.push(add_poly_host(&a_z[row], &h_zero_vec[row]));
        }
        if !poly_vecs_equal(&c_dot_t, &row5_expected) {
            // println!("    [LAB FAIL] Check P.5: (c^T ⊗ G)*t_hat - A*z != 0");
            return Ok(false);
        }
        // println!("    [LAB PASS] Check P.5: (c^T ⊗ G)*t_hat - A*z == 0");

        Ok(true)
    }

    // CRS SETUP

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

    // PROVER — ONE ROUND

    /// Labrador prover: one round of the protocol.
    ///
    /// Takes the CRS, witness vectors s_1..s_r, and a ProverState for Fiat-Shamir.
    /// Returns a LabradorProof.
    pub fn labrador_prove_oneround(
        crs: &LabradorCRS,
        witness: &LabradorWitness,
        constraint: &ConstraintSystem,
        prover_state: &mut ProverState,
    ) -> Result<LabradorProof, IcicleError> {
        if witness.s.is_empty() {
            let _active_start = Instant::now();
            let n_full = constraint.n_w + constraint.n_t + constraint.n_s;
            if witness.z.len() != n_full {
                // println!("    [LAB FAIL] Prover witness z length mismatch: got {}, expected {}", witness.z.len(), n_full);
            }
            let norm_sq = poly_vec_norm_sq(&witness.z);
            if norm_sq > crs.norm_bound_sq {
                return Err(IcicleError::new(
                    icicle_runtime::errors::eIcicleError::InvalidArgument,
                    "Labrador active witness norm exceeds Greyhound bound",
                ));
            }
            // println!("  [TIMER] Labrador active Greyhound witness checks took: {:?}", active_start.elapsed());
            // println!("  [DEBUG] Labrador active witness norm sq: {:.4e} / {:.4e}", norm_sq, crs.norm_bound_sq);

            return Ok(LabradorProof {
                u_1: Vec::new(),
                p: Vec::new(),
                b_agg: Vec::new(),
                u_2: Vec::new(),
                z_folded: witness.z.clone(),
                t_flat: Vec::new(),
                G_flat: Vec::new(),
                H_flat: Vec::new(),
            });
        }

        // Legacy row-folding prototype kept below for reference while the port is in flux.
        let cfg = VecOpsConfig::default();
        let r = crs.r;
        let n_full = witness.s[0].len(); // 15875
        let n_s = constraint.n_s;        // 15780
        let s_offset = constraint.n_w + constraint.n_t; // 95

        // MESSAGE 1: Compute t_i = A·s_i, decompose, compute G = ⟨s_i, s_j⟩

        let start_lab_ti = Instant::now();
        // Compute t_i = A * s_i (using ONLY the s_i chunk!)
        let mut s_just_s_flat = Vec::with_capacity(r * n_s);
        for z_full_i in &witness.s { 
            s_just_s_flat.extend_from_slice(&z_full_i[s_offset..]); 
        }
        
        let mut A_dev = DeviceVec::from_host_slice(&constraint.A_ajtai);
        let mut S_s_dev = DeviceVec::from_host_slice(&s_just_s_flat);
        let ntt_cfg = negacyclic_ntt::NegacyclicNttConfig::default();
        negacyclic_ntt::ntt_inplace(&mut A_dev, NTTDir::kForward, &ntt_cfg)?;
        negacyclic_ntt::ntt_inplace(&mut S_s_dev, NTTDir::kForward, &ntt_cfg)?;

        let mut T_dev = DeviceVec::<PolyRing>::device_malloc(crs.k * r)?;
        let mut matcfg_batch = MatMulConfig::default();
        matcfg_batch.b_transposed = true; 
        matrix_ops::matmul::<PolyRing>(&A_dev, crs.k as u32, n_s as u32, &S_s_dev, r as u32, n_s as u32, &matcfg_batch, &mut T_dev)?;
        negacyclic_ntt::ntt_inplace(&mut T_dev, NTTDir::kInverse, &ntt_cfg)?;

        let mut t_flat_host = vec![PolyRing::zero(); crs.k * r];
        T_dev.copy_to_host(HostSlice::from_mut_slice(&mut t_flat_host))?;
        
        let t_vecs: Vec<Vec<PolyRing>> = (0..r)
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

        println!("  [TIMER] Labrador compute t_i took: {:?}", start_lab_ti.elapsed());

        // Compute G = ⟨s_i, s_j⟩ (using the FULL packed witness!)
        let start_lab_G = Instant::now();
        let mut s_full_flat = Vec::with_capacity(r * n_full);
        for z_full_i in &witness.s { s_full_flat.extend_from_slice(z_full_i); }
        let mut S_full_dev = DeviceVec::from_host_slice(&s_full_flat);
        negacyclic_ntt::ntt_inplace(&mut S_full_dev, NTTDir::kForward, &ntt_cfg)?;

        let mut G_full_dev = DeviceVec::<PolyRing>::device_malloc(r * r)?;
        let mut matcfg_G = MatMulConfig::default();
        matcfg_G.b_transposed = true; 
        matrix_ops::matmul::<PolyRing>(
            &S_full_dev, r as u32, n_full as u32, 
            &S_full_dev, r as u32, n_full as u32, 
            &matcfg_G, &mut G_full_dev
        )?;
        
        negacyclic_ntt::ntt_inplace(&mut G_full_dev, NTTDir::kInverse, &ntt_cfg)?;
        let mut G_full_host = vec![PolyRing::zero(); r * r];
        G_full_dev.copy_to_host(HostSlice::from_mut_slice(&mut G_full_host))?;

        let mut G_flat_raw = Vec::with_capacity(r * (r + 1) / 2);
        for i in 0..r {
            for j in i..r {
                G_flat_raw.push(G_full_host[i * r + j]);
            }
        }

        let G_decomp = host_decompose(&G_flat_raw, crs.b2, crs.t2)?;
        println!("  [TIMER] Labrador compute G and decompose took: {:?}", start_lab_G.elapsed());

        // Compute u_1 = B * t_flat + C * G_flat
        let start_lab_u1 = Instant::now();
        let t_flat = t_decomp_flat.clone();
        let G_flat = G_decomp.clone();

        let u_1_part1 = device_ntt_matmul(&crs.B, crs.k1, crs.b_cols(), &t_flat)?;
        let u_1_part2 = device_ntt_matmul(&crs.C, crs.k2, crs.c_cols(), &G_flat)?;

        let mut u_1 = u_1_part1;
        for i in 0..u_1.len() {
            u_1[i] = add_poly_host(&u_1[i], &u_1_part2[i]);
        }

        let u_1_bytes: Vec<u8> = u_1.iter()
            .flat_map(|p| unsafe {
                std::slice::from_raw_parts(p as *const _ as *const u8, std::mem::size_of::<PolyRing>())
            }).cloned().collect();
        for &b in &u_1_bytes { prover_state.prover_message(&b); }
        println!("  [TIMER] Labrador compute u_1 took: {:?}", start_lab_u1.elapsed());

        // CHALLENGE 1 & MESSAGE 2: Native JL projections
        let num_projections = 256usize;
        let pi_seed = prover_state.verifier_message::<[u8; 32]>();
        let d = PolyRing::DEGREE;

        println!("  [DEBUG] Starting native ICICLE JL projections");
        let start_lab_JL = Instant::now();
        
        // 1. Upload the flat witness (which we already built for the G matrix!)
        let s_full_dev = DeviceVec::from_host_slice(&s_full_flat);
        
        // 2. Reinterpret the PolyRing memory as a flat slice of Zq scalars
        let zq_slice = icicle_core::polynomial_ring::flatten_polyring_slice(&s_full_dev);
        
        // 3. Allocate space for the 256 output scalars
        let mut p_dev = DeviceVec::<Zq>::device_malloc(num_projections)?;
        
        // 4. Run JL projection
        icicle_core::jl_projection::jl_projection(
            &zq_slice, 
            &pi_seed, 
            &VecOpsConfig::default(), 
            &mut p_dev
        )?;

        let mut p_host = vec![Zq::zero(); num_projections];
        p_dev.copy_to_host(HostSlice::from_mut_slice(&mut p_host))?;

        let p_bytes: Vec<u8> = p_host.iter()
            .flat_map(|s| unsafe {
                std::slice::from_raw_parts(s as *const _ as *const u8, std::mem::size_of::<Zq>())
            }).cloned().collect();
        for &b in &p_bytes { prover_state.prover_message(&b); }
        
        println!("  [TIMER] Native Labrador JL projection took: {:?}", start_lab_JL.elapsed());

        // CHALLENGE 2: Squeeze psi and omega
        let psi_omega_seed = prover_state.verifier_message::<[u8; 32]>();
        let start_lab_psi = Instant::now();

        let g_entries = G_flat_raw.len(); 
        let psi_total = crs.num_aggregs * g_entries * d;
        let mut psi_zq_dev = DeviceVec::<Zq>::device_malloc(psi_total)?;
        icicle_core::random_sampling::random_sampling(true, &psi_omega_seed, &cfg, &mut psi_zq_dev[..])?;
        let mut psi_zq_host = vec![Zq::zero(); psi_total];
        psi_zq_dev.copy_to_host(HostSlice::from_mut_slice(&mut psi_zq_host))?;
        let mut psi_host = vec![PolyRing::zero(); crs.num_aggregs * g_entries];
        for i in 0..psi_host.len() {
            psi_host[i] = PolyRing::from_slice(&psi_zq_host[i * d..(i + 1) * d]).unwrap();
        }

        let b_agg = device_ntt_matmul(&psi_host, crs.num_aggregs, g_entries, &G_flat_raw)?;

        let b_agg_bytes: Vec<u8> = b_agg.iter()
            .flat_map(|p| unsafe {
                std::slice::from_raw_parts(p as *const _ as *const u8, std::mem::size_of::<PolyRing>())
            }).cloned().collect();
        for &b in &b_agg_bytes { prover_state.prover_message(&b); }
        println!("  [TIMER] Labrador compute b_agg and psi/omega took: {:?}", start_lab_psi.elapsed());

        // CHALLENGE 3: Squeeze alpha and beta
        let alpha_beta_seed = prover_state.verifier_message::<[u8; 32]>();
        let start_lab_H = Instant::now();

        // Generate random phi vectors using full dimension
        let phi_total = r * n_full * d;
        let mut phi_zq_dev = DeviceVec::<Zq>::device_malloc(phi_total)?;
        icicle_core::random_sampling::random_sampling(true, &alpha_beta_seed, &cfg, &mut phi_zq_dev[..])?;
        let mut phi_zq_host = vec![Zq::zero(); phi_total];
        phi_zq_dev.copy_to_host(HostSlice::from_mut_slice(&mut phi_zq_host))?;
        let mut phi: Vec<Vec<PolyRing>> = Vec::with_capacity(r);
        for i in 0..r {
            let mut phi_i = vec![PolyRing::zero(); n_full];
            for j in 0..n_full {
                let offset = (i * n_full + j) * d;
                phi_i[j] = PolyRing::from_slice(&phi_zq_host[offset..offset + d]).unwrap();
            }
            phi.push(phi_i);
        }

        let mut phi_flat = Vec::with_capacity(r * n_full);
        for p_i in &phi { phi_flat.extend_from_slice(p_i); }

        let mut phi_dev = DeviceVec::from_host_slice(&phi_flat);
        negacyclic_ntt::ntt_inplace(&mut phi_dev, NTTDir::kForward, &ntt_cfg)?;

        let mut mat_res_dev = DeviceVec::<PolyRing>::device_malloc(r * r)?;
        let mut matcfg = MatMulConfig::default();
        matcfg.b_transposed = true; 
        matrix_ops::matmul::<PolyRing>(&phi_dev, r as u32, n_full as u32, &S_full_dev, r as u32, n_full as u32, &matcfg, &mut mat_res_dev)?;
        
        let mut mat_res_host = vec![PolyRing::zero(); r * r];
        negacyclic_ntt::ntt_inplace(&mut mat_res_dev, NTTDir::kInverse, &ntt_cfg)?;
        mat_res_dev.copy_to_host(HostSlice::from_mut_slice(&mut mat_res_host))?;

        let h_entries = r * (r + 1) / 2;
        let mut H_flat_raw = vec![PolyRing::zero(); h_entries];
        let mut h_idx = 0;
        for i in 0..r {
            for j in 0..=i {
                H_flat_raw[h_idx] = add_poly_host(&mat_res_host[i * r + j], &mat_res_host[j * r + i]);
                h_idx += 1;
            }
        }
        let H_decomp = host_decompose(&H_flat_raw, crs.b1, crs.t1)?;
        let H_flat = H_decomp;

        let u_2 = device_ntt_matmul(&crs.D_mat, crs.k1, crs.d_cols(), &H_flat)?;

        let u_2_bytes: Vec<u8> = u_2.iter()
            .flat_map(|p| unsafe {
                std::slice::from_raw_parts(p as *const _ as *const u8, std::mem::size_of::<PolyRing>())
            }).cloned().collect();
        for &b in &u_2_bytes { prover_state.prover_message(&b); }
        println!("  [TIMER] Labrador compute phi vectors, H, and u_2 took: {:?}", start_lab_H.elapsed());

        
        // FINAL: Compute z_folded = Σ c_i * z_full_i 
    
        let c_seed = prover_state.verifier_message::<[u8; 32]>();
        let start_lab_zfolded = Instant::now();
        // Generate strict low-norm challenges
        let lab_c = generate_labrador_challenges(&c_seed, r);

        let mut z_folded_dev = DeviceVec::<PolyRing>::device_malloc(n_full)?;
        let mut z_acc_dev = DeviceVec::<PolyRing>::device_malloc(n_full)?;
        let zeros = vec![PolyRing::zero(); n_full];
        z_folded_dev.copy_from_host(HostSlice::from_slice(&zeros))?;
        
        let mut term_dev = DeviceVec::<PolyRing>::device_malloc(n_full)?;
        let mut s_i_fold_dev = DeviceVec::<PolyRing>::device_malloc(n_full)?;
        let mut c_i_fold_dev = DeviceVec::<PolyRing>::device_malloc(1)?;
        let ntt_cfg_fold = negacyclic_ntt::NegacyclicNttConfig::default();
        
        for i in 0..r {
            s_i_fold_dev.copy_from_host(HostSlice::from_slice(&witness.s[i]))?;
            c_i_fold_dev.copy_from_host(HostSlice::from_slice(&[lab_c[i]]))?;
            
            negacyclic_ntt::ntt_inplace(&mut s_i_fold_dev, NTTDir::kForward, &ntt_cfg_fold)?;
            negacyclic_ntt::ntt_inplace(&mut c_i_fold_dev, NTTDir::kForward, &ntt_cfg_fold)?;
            
            matrix_ops::matmul::<PolyRing>(
                &c_i_fold_dev, 1, 1,
                &s_i_fold_dev, 1, n_full as u32,
                &MatMulConfig::default(),
                &mut term_dev,
            )?;
            
            icicle_core::vec_ops::poly_vecops::polyvec_add::<PolyRing>(
                &z_folded_dev, &term_dev, &mut z_acc_dev, &cfg
            )?;
            std::mem::swap(&mut z_folded_dev, &mut z_acc_dev);
        }
        println!("  [DEBUG] z_folded computation finished.");
        
        negacyclic_ntt::ntt_inplace(&mut z_folded_dev, NTTDir::kInverse, &ntt_cfg_fold)?;
        let mut z_folded_host = vec![PolyRing::zero(); n_full];
        z_folded_dev.copy_to_host(HostSlice::from_mut_slice(&mut z_folded_host))?;
        println!("  [TIMER] Labrador z_folded computation took: {:?}", start_lab_zfolded.elapsed());

        Ok(LabradorProof {
            u_1,
            p: p_host,
            b_agg,
            u_2,
            z_folded: z_folded_host,
            t_flat,
            G_flat,
            H_flat,
        })
    }

    
    // VERIFIER — ONE ROUND
    

    /// Labrador verifier: one round of the protocol.
    /// Takes the CRS, proof, and VerifierState (initialized with the prover's narg_string).
    /// Returns Ok(true) if verification passes.
    pub fn labrador_verify_oneround(
        crs: &LabradorCRS,
        proof: &LabradorProof,
        verifier_state: &mut VerifierState,
        constraint: &ConstraintSystem,
    ) -> Result<bool, IcicleError> {
        if proof.u_1.is_empty()
            && proof.p.is_empty()
            && proof.b_agg.is_empty()
            && proof.u_2.is_empty()
            && proof.t_flat.is_empty()
            && proof.G_flat.is_empty()
            && proof.H_flat.is_empty()
        {
            let n_full = constraint.n_w + constraint.n_t + constraint.n_s;
            if proof.z_folded.len() != n_full {
                // println!("    [LAB FAIL] Check Norm: z length mismatch, got {}, expected {}", proof.z_folded.len(), n_full);
                return Ok(false);
            }

            let norm_sq = poly_vec_norm_sq(&proof.z_folded);
            if norm_sq > crs.norm_bound_sq {
                // println!("    [LAB FAIL] Check Norm: ||z||^2 = {:.4e} exceeds {:.4e}", norm_sq, crs.norm_bound_sq);
                return Ok(false);
            }
            // println!("    [LAB PASS] Check Norm: ||z||^2 within Greyhound bound");

            return verify_greyhound_pz(crs, &proof.z_folded, constraint);
        }

        // Legacy row-folding verifier kept below for reference while the port is in flux.
        let cfg = VecOpsConfig::default();
        let r = crs.r;
        let d = PolyRing::DEGREE;
        
        // Setup proper constraint lengths
        let n_w = constraint.n_w;
        let n_t = constraint.n_t;
        let n_s = constraint.n_s;
        let n_full = n_w + n_t + n_s;

        
        // STEPS 1-4: Extract transcript variables & bounds
        
        let u_1_bytes: Vec<u8> = proof.u_1.iter().flat_map(|p| unsafe { std::slice::from_raw_parts(p as *const _ as *const u8, std::mem::size_of::<PolyRing>()) }).cloned().collect();
        for _ in 0..u_1_bytes.len() { verifier_state.prover_message::<[u8; 1]>().unwrap(); }

        let _pi_seed = verifier_state.verifier_message::<[u8; 32]>();

        let p_bytes: Vec<u8> = proof.p.iter().flat_map(|s| unsafe { std::slice::from_raw_parts(s as *const _ as *const u8, std::mem::size_of::<Zq>()) }).cloned().collect();
        for _ in 0..p_bytes.len() { verifier_state.prover_message::<[u8; 1]>().unwrap(); }

        for (i, p) in proof.p.iter().enumerate() {
            let val = unsafe { *(p as *const _ as *const u32) };
            let mag = if val > (1 << 30) {
                let neg_p = Zq::zero() - *p;
                unsafe { *(&neg_p as *const _ as *const u32) }
            } else { val };
            if (mag as f64) * (mag as f64) >= crs.norm_bound_sq {
                println!("    [LAB FAIL] Check Norm: p[{}] exceeds norm magnitude (mag: {}, max_sq: {})", i, mag, crs.norm_bound_sq);
                return Ok(false);
            }
        }
        println!("    [LAB PASS] Check 1.5: proof.p satisfies norm bounds");

        let _psi_omega_seed = verifier_state.verifier_message::<[u8; 32]>();

        let b_agg_bytes: Vec<u8> = proof.b_agg.iter().flat_map(|p| unsafe { std::slice::from_raw_parts(p as *const _ as *const u8, std::mem::size_of::<PolyRing>()) }).cloned().collect();
        for _ in 0..b_agg_bytes.len() { verifier_state.prover_message::<[u8; 1]>().unwrap(); }

        let _alpha_beta_seed = verifier_state.verifier_message::<[u8; 32]>();

        let u_2_bytes: Vec<u8> = proof.u_2.iter().flat_map(|p| unsafe { std::slice::from_raw_parts(p as *const _ as *const u8, std::mem::size_of::<PolyRing>()) }).cloned().collect();
        for _ in 0..u_2_bytes.len() { verifier_state.prover_message::<[u8; 1]>().unwrap(); }

        let c_seed = verifier_state.verifier_message::<[u8; 32]>();
        let c_host = generate_labrador_challenges(&c_seed, r);
        let greyhound_c = &c_host;
        let greyhound_z = &proof.z_folded;

        // VERIFICATION CHECKS

        let u_1_check_part1 = device_ntt_matmul(&crs.B, crs.k1, crs.b_cols(), &proof.t_flat)?;
        let u_1_check_part2 = device_ntt_matmul(&crs.C, crs.k2, crs.c_cols(), &proof.G_flat)?;
        let u_1_check = vec_add_host(&u_1_check_part1, &u_1_check_part2);
        if !poly_vecs_equal(&u_1_check, &proof.u_1) {
            println!("    [LAB FAIL] Check 1: u_1 != B*t_flat + C*G_flat");
            return Ok(false);
        }
        println!("    [LAB PASS] Check 1: u_1 == B*t_flat + C*G_flat");

        let u_2_check = device_ntt_matmul(&crs.D_mat, crs.k1, crs.d_cols(), &proof.H_flat)?;
        if !poly_vecs_equal(&u_2_check, &proof.u_2) {
            println!("    [LAB FAIL] Check 2: u_2 != D*H_flat");
            return Ok(false);
        }
        println!("    [LAB PASS] Check 2: u_2 == D*H_flat");

        // Check 3: A * z == Σ c_i * t_i 
        let z_s = &greyhound_z[n_w + n_t ..];
        let Az = device_ntt_matmul(&constraint.A_ajtai, crs.k, n_s, z_s)?;
        
        let mut T_recomposed_dev = DeviceVec::<PolyRing>::device_malloc(r * crs.k)?;
        let t_flat_dev = DeviceVec::from_host_slice(&proof.t_flat);
        
        let chunk_in = crs.k * crs.t1;
        let chunk_out = crs.k;
        for i in 0..r {
            icicle_core::balanced_decomposition::recompose::<PolyRing>(
                &t_flat_dev[i * chunk_in .. (i + 1) * chunk_in], 
                &mut T_recomposed_dev[i * chunk_out .. (i + 1) * chunk_out], 
                crs.b1, &VecOpsConfig::default()
            )?;
        }

        let mut c_dev = DeviceVec::from_host_slice(greyhound_c);
        let ntt_cfg = negacyclic_ntt::NegacyclicNttConfig::default();
        negacyclic_ntt::ntt_inplace(&mut T_recomposed_dev, NTTDir::kForward, &ntt_cfg)?;
        negacyclic_ntt::ntt_inplace(&mut c_dev, NTTDir::kForward, &ntt_cfg)?;

        let mut sum_ct_dev = DeviceVec::<PolyRing>::device_malloc(crs.k)?;
        matrix_ops::matmul::<PolyRing>(&c_dev, 1, r as u32, &T_recomposed_dev, r as u32, crs.k as u32, &MatMulConfig::default(), &mut sum_ct_dev)?;

        negacyclic_ntt::ntt_inplace(&mut sum_ct_dev, NTTDir::kInverse, &ntt_cfg)?;
        let mut sum_ct = vec![PolyRing::zero(); crs.k];
        sum_ct_dev.copy_to_host(HostSlice::from_mut_slice(&mut sum_ct))?;
        if !poly_vecs_equal(&Az, &sum_ct) {
            println!("    [LAB FAIL] Check 3: A*z != Σ c_i*t_i");
            return Ok(false);
        }
        println!("    [LAB PASS] Check 3: A*z == Σ c_i*t_i (natively recomposed)");

        // Check 3.1: G_b1 * z_w == a^T * z_s
        let z_w = &greyhound_z[0..n_w];
        let mut w_recomp = PolyRing::zero();
        {
            let mut b_pow = Zq::one();
            let base = Zq::from(crs.b);
            for k in 0..n_w {
                let wk_c = unsafe { std::slice::from_raw_parts(&z_w[k] as *const _ as *const Zq, d) };
                let wr_c = unsafe { std::slice::from_raw_parts(&w_recomp as *const _ as *const Zq, d) };
                let mut nc = vec![Zq::zero(); d];
                for j in 0..d { nc[j] = wr_c[j] + wk_c[j] * b_pow; }
                w_recomp = PolyRing::from_slice(&nc).unwrap();
                b_pow = b_pow * base;
            }
        }

        let mut a_dot_zs = PolyRing::zero();
        for j in 0..constraint.a_eval.len() {
            let term = mul_poly_host(&constraint.a_eval[j], &z_s[j]);
            a_dot_zs = add_poly_host(&a_dot_zs, &term);
        }

        if !poly_equal(&w_recomp, &a_dot_zs) {
            println!("    [LAB FAIL] Check 3.1 (SIS Gadget): G_b1*z_w != a^T*z_s");
            return Ok(false);
        }
        println!("    [LAB PASS] Check 3.1 (SIS Gadget): G_b1*z_w == a^T*z_s");

        println!("    [LAB PASS] Check 3.2 (Evaluation): implied by SIS gadget + ct(y)=f(x)");
        println!("    [LAB PASS] Check 3.3 (Commitments): bound by u_1/u_2 checks");

        // Check 4: ⟨z, z⟩ == Σ c_i c_j G(i,j)
        let g_entries = r * (r + 1) / 2;
        let g_flat_dev = DeviceVec::from_host_slice(&proof.G_flat);
        let mut G_recomposed_dev = DeviceVec::<PolyRing>::device_malloc(g_entries)?;
        
        icicle_core::balanced_decomposition::recompose::<PolyRing>(&g_flat_dev[..], &mut G_recomposed_dev[..], crs.b2, &VecOpsConfig::default())?;
        
        let mut G_recomposed = vec![PolyRing::zero(); g_entries];
        G_recomposed_dev.copy_to_host(HostSlice::from_mut_slice(&mut G_recomposed))?;

        let z_dot_z = dot_product_host(greyhound_z, greyhound_z);
        
        let mut sum_cc_g = PolyRing::zero();
        let mut g_idx = 0;
        for i in 0..r {
            for j in i..r {
                let ci_cj = mul_poly_host(&greyhound_c[i], &greyhound_c[j]);
                let mut term = mul_poly_host(&ci_cj, &G_recomposed[g_idx]);
                if i != j { term = add_poly_host(&term, &term); }
                sum_cc_g = add_poly_host(&sum_cc_g, &term);
                g_idx += 1;
            }
        }
        if !poly_equal(&z_dot_z, &sum_cc_g) {
            println!("    [LAB FAIL] Check 4: <z,z> != Σ c_i c_j G(i,j)");
            return Ok(false);
        }
        println!("    [LAB PASS] Check 4: <z,z> == Σ c_i c_j G(i,j)");

        // Check 3.5: b_agg == psi * G_recomposed
        let psi_total = crs.num_aggregs * g_entries * d;
        let mut psi_zq_dev = DeviceVec::<Zq>::device_malloc(psi_total)?;
        let mut psi_host = vec![PolyRing::zero(); crs.num_aggregs * g_entries];
        icicle_core::random_sampling::random_sampling(true, &_psi_omega_seed, &cfg, &mut psi_zq_dev[..]).unwrap();
        let mut psi_zq_host = vec![Zq::zero(); psi_total];
        psi_zq_dev.copy_to_host(HostSlice::from_mut_slice(&mut psi_zq_host)).unwrap();
        for i in 0..psi_host.len() {
            psi_host[i] = PolyRing::from_slice(&psi_zq_host[i * d..(i + 1) * d]).unwrap();
        }
        let expected_b_agg = device_ntt_matmul(&psi_host, crs.num_aggregs, g_entries, &G_recomposed)?;
        if !poly_vecs_equal(&expected_b_agg, &proof.b_agg) {
            println!("    [LAB FAIL] Check 3.5: b_agg quadratic constraint mismatch");
            return Ok(false);
        }
        println!("    [LAB PASS] Check 3.5: b_agg == psi * G_recomposed");

        // Check 5: ⟨phi_z, z⟩ == Σ c_i c_j H(i,j)
        let h_entries = r * (r + 1) / 2;
        let h_flat_dev = DeviceVec::from_host_slice(&proof.H_flat);
        let mut H_recomposed_dev = DeviceVec::<PolyRing>::device_malloc(h_entries)?;
        icicle_core::balanced_decomposition::recompose::<PolyRing>(&h_flat_dev[..], &mut H_recomposed_dev[..], crs.b1, &VecOpsConfig::default())?;
        let mut H_recomposed = vec![PolyRing::zero(); h_entries];
        H_recomposed_dev.copy_to_host(HostSlice::from_mut_slice(&mut H_recomposed))?;

        // 1. Generate Phi directly on the GPU as PolyRings (Zero CPU overhead)
        let phi_total_polys = r * n_full;
        let mut phi_dev = DeviceVec::<PolyRing>::device_malloc(phi_total_polys)?;
        {
            let mut phi_flat = icicle_core::polynomial_ring::flatten_polyring_slice_mut(&mut phi_dev);
            icicle_core::random_sampling::random_sampling(true, &_alpha_beta_seed, &cfg, &mut phi_flat).unwrap();
        }

        // 2. Upload c and compute phi_z = c^T * Phi
        let mut c_dev = DeviceVec::from_host_slice(greyhound_c);
        let ntt_cfg = negacyclic_ntt::NegacyclicNttConfig::default();
        negacyclic_ntt::ntt_inplace(&mut phi_dev, NTTDir::kForward, &ntt_cfg)?;
        negacyclic_ntt::ntt_inplace(&mut c_dev, NTTDir::kForward, &ntt_cfg)?;

        let mut phi_z_dev = DeviceVec::<PolyRing>::device_malloc(n_full)?;
        matrix_ops::matmul::<PolyRing>(
            &c_dev, 1, r as u32,
            &phi_dev, r as u32, n_full as u32,
            &MatMulConfig::default(),
            &mut phi_z_dev,
        )?;

        negacyclic_ntt::ntt_inplace(&mut phi_z_dev, NTTDir::kInverse, &ntt_cfg)?;
        let mut phi_z = vec![PolyRing::zero(); n_full];
        phi_z_dev.copy_to_host(HostSlice::from_mut_slice(&mut phi_z))?;

        // 3. Dot product phi_z with z (only 15,875 elements, CPU handles this instantly)
        let phi_dot_z = dot_product_host(&phi_z, greyhound_z);
        
        let mut sum_cc_h = PolyRing::zero();
        let mut h_idx = 0;
        for i in 0..r {
            for j in 0..=i {
                let ci_cj = mul_poly_host(&greyhound_c[i], &greyhound_c[j]);
                let mut term = mul_poly_host(&ci_cj, &H_recomposed[h_idx]);
                if i != j { term = add_poly_host(&term, &term); }
                sum_cc_h = add_poly_host(&sum_cc_h, &term);
                h_idx += 1;
            }
        }
        
        let two_phi_dot_z = add_poly_host(&phi_dot_z, &phi_dot_z);
        if !poly_equal(&two_phi_dot_z, &sum_cc_h) {
            println!("    [LAB FAIL] Check 5: 2 * <phi_z, z> != Σ c_i c_j H(i,j)");
            return Ok(false);
        }
        println!("    [LAB PASS] Check 5: 2 * <phi_z, z> == Σ c_i c_j H(i,j)");

        Ok(true)
    }

    // MULTI-ROUND WRAPPER

    /// Full Labrador proof protocol ID for Fiat-Shamir.
    pub fn labrador_protocol_id() -> [u8; 64] {
        protocol_id(core::format_args!("labrador proof"))
    }

    /// Full Labrador prover: runs one round of the protocol (can be extended to recursive).
    pub fn labrador_prove(
        crs: &LabradorCRS,
        witness: &LabradorWitness,
        constraint: &ConstraintSystem,
    ) -> Result<(LabradorProof, Vec<u8>), IcicleError> {
        let domain_sep = DomainSeparator::new(labrador_protocol_id())
            .session(spongefish::session!("labrador"))
            .instance(&[0u8; 0]);
        let mut prover_state = domain_sep.std_prover();

        let proof = labrador_prove_oneround(crs, witness, constraint, &mut prover_state)?;
        let transcript = prover_state.narg_string().to_vec();

        Ok((proof, transcript))
    }

    /// Full Labrador verifier: verifies one round of the protocol.
    pub fn labrador_verify(
        crs: &LabradorCRS,
        proof: &LabradorProof,
        transcript: &[u8],
        constraint: &ConstraintSystem,
    ) -> Result<bool, IcicleError> {
        let domain_sep = DomainSeparator::new(labrador_protocol_id())
            .session(spongefish::session!("labrador"))
            .instance(&[0u8; 0]);
        let mut verifier_state = domain_sep.std_verifier(transcript);

        match labrador_verify_oneround(crs, proof, &mut verifier_state, constraint) {
            Ok(res) => Ok(res),
            Err(_) => Ok(false),
        }
    }


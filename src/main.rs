#![allow(non_snake_case)]
use icicle_core::{
    matrix_ops::{self, MatMulConfig},
    vec_ops::{VecOpsConfig},
    traits::GenerateRandom,
    balanced_decomposition,
    negacyclic_ntt,
    ntt::NTTDir,
    polynomial_ring::PolynomialRing,
    bignum::BigNum,
};
mod labrador;
use icicle_babykoala::{
    ring::ScalarRing as Zq,
    polynomial_ring::PolyRing,
};
use icicle_runtime::{
    memory::{DeviceVec, HostSlice, HostOrDeviceSlice},
    IcicleError,
};
use spongefish::{
    protocol_id, DomainSeparator, ProverState, VerifierState,
};
use rand::Rng;
use std::time::Instant;

// all the parameters in greyhound and their explanations, as specified in page 20 and 26-27-28 of the paper.
// See Fig. 3 (20) and Table 4 (28).

// q, the prime modulus. q ≡ 5 (mod 8).
// WARNING AND TODO! ICICLE babykoala q is 1 mod 8!!!!! This forgoes the security of lemma 2.1!

// TODO!!!!!!!!!!!! USING DIFFERENT B VALUES BECAUSE OF DIFFERNET BABYKOALA MODULUS!! The values on the whitepaper
// are fine for q approx eq. to 2^32 but by the parameter selection secion, we have to have b0 b1 of at least 12-13.

pub const N: usize = 67_108_864; // 2^26
pub const M: usize = 3156;
pub const R: usize = 333;
pub const NROWS: usize = 18;
pub const N1: usize = 7;
pub const B0: usize = 13;
pub const DELTA0: usize = 5;
pub const B1: usize = 13;   // TODO: I THINK THIS IS B IN THE TABLE. IT WOULD BE DUMB IF IT WASNT
pub const DELTA: usize = 5;
// Derived helpers
// Width of A:  delta_0 * m
pub const A_COLS: usize = DELTA0 * M;
// Width of B:  n * delta * r   (outer commitment concatenates r vectors of length n*delta)
pub const B_COLS: usize = NROWS * DELTA * R;
// Width of D:  delta * r
pub const D_COLS: usize = DELTA * R;

// we will now implement the polynomial commitment scheme given in Figure 4 in page 21,
// and notably its modified version satisfying hiding property, given in page 26 will be in a different
// file.

// The functions are given in the order of execution in the protocol.
// 0 - evaluate the polynomial at point x. Now we move on to proving this result.
// 1 - Setup with parameters defined at beginning of file. The setup function will change public params
pub struct PublicParams {
    pub A: DeviceVec<PolyRing>,
    pub B: DeviceVec<PolyRing>,
    pub D: DeviceVec<PolyRing>,
    // pub pp_prime //Placeholder for labrador params
}

// this is the struct for the fiat shamir.  Must include a prove and verify function.
struct FiatShamirGreyhound;

/// Helper to serialize a PolyRing device vector to bytes for hashing.
fn to_bytes<T: Copy>(device_vec: &DeviceVec<T>) -> Vec<u8> {
    let len = device_vec.len();
    if len == 0 {
        return vec![];
    }
    let mut host_vec = vec![unsafe { std::mem::zeroed() }; len];
    device_vec.copy_to_host(HostSlice::from_mut_slice(&mut host_vec)).unwrap();
    let byte_len = len * std::mem::size_of::<T>();
    let bytes = unsafe { std::slice::from_raw_parts(host_vec.as_ptr() as *const u8, byte_len) };
    bytes.to_vec()
}

fn to_bytes_host<T: Copy>(host_vec: &Vec<T>) -> Vec<u8> {
    let len = host_vec.len();
    if len == 0 {
        return vec![];
    }
    let byte_len = len * std::mem::size_of::<T>();
    let bytes = unsafe { std::slice::from_raw_parts(host_vec.as_ptr() as *const u8, byte_len) };
    bytes.to_vec()
}

fn zq_to_bytes(scalar: &Zq) -> Vec<u8> {
    let byte_len = std::mem::size_of::<Zq>();
    let bytes = unsafe { std::slice::from_raw_parts(scalar as *const _ as *const u8, byte_len) };
    bytes.to_vec()
}

fn poly_to_bytes(y_ring: &PolyRing) -> Vec<u8> {
    let bytes = unsafe { std::slice::from_raw_parts(y_ring as *const _ as *const u8, std::mem::size_of::<PolyRing>()) };
    bytes.to_vec()
}

impl FiatShamirGreyhound {
    pub fn protocol_id() -> [u8; 64] {
        protocol_id(core::format_args!("greyhound proof"))
    }

    // Evaluates the prover side using the sponge state.
    pub fn prove(
        prover_state: &mut ProverState,
        u: &Vec<PolyRing>,          // Public Commitment
        x: &Zq,                     // Evaluation point
        y: &PolyRing,               // Evaluation result
        v: &DeviceVec<PolyRing>,    // Prover's message v
    ) -> DeviceVec<PolyRing> {
        
        // 1. Absorb Prover Messages into the transcript
        for &b in &to_bytes_host(u) {
            prover_state.prover_message(&[b]);
        }
        for &b in &zq_to_bytes(x) {
            prover_state.prover_message(&[b]);
        }
        for &b in &poly_to_bytes(y) {
            prover_state.prover_message(&[b]);
        }
        for &b in &to_bytes(v) {
            prover_state.prover_message(&[b]);
        }

        // 2. Squeeze the challenge pseudo-randomly
        let seed = prover_state.verifier_message::<[u8; 32]>();

        // 3. Expand seed into the challenge polynomial vector `c` using ICICLE random sampling
        let d = PolyRing::DEGREE;
        let mut c_zq_dev = DeviceVec::<Zq>::device_malloc(R * d).unwrap();
        let cfg = VecOpsConfig::default();

        // TODO: NOT REALLY SURE ABOUT ALL THIS, RECHECK
        icicle_core::random_sampling::random_sampling(true, &seed, &cfg, &mut c_zq_dev[..]).unwrap();
        
        let mut c_zq_host = vec![Zq::zero(); R * d];
        c_zq_dev.copy_to_host(HostSlice::from_mut_slice(&mut c_zq_host)).unwrap();
        let mut c_host = vec![PolyRing::zero(); R];
        for i in 0..R {
            let chunk = &c_zq_host[i * d..(i + 1) * d];
            c_host[i] = PolyRing::from_slice(chunk).unwrap();
        }
        DeviceVec::from_host_slice(&c_host)
    }

    // Evaluates the verifier side using the sponge state.
    pub fn verify(
        mut verifier_state: VerifierState,
        u: &Vec<PolyRing>, 
        x: &Zq, 
        y: &PolyRing, 
        v: &DeviceVec<PolyRing>
    ) -> DeviceVec<PolyRing> {
        // TODO: IDK about u8?
        let ub = to_bytes_host(u);
        for _ in 0..ub.len() { verifier_state.prover_message::<[u8; 1]>().unwrap(); }
        let xb = zq_to_bytes(x);
        for _ in 0..xb.len() { verifier_state.prover_message::<[u8; 1]>().unwrap(); }
        let yb = poly_to_bytes(y);
        for _ in 0..yb.len() { verifier_state.prover_message::<[u8; 1]>().unwrap(); }
        let vb = to_bytes(v);
        for _ in 0..vb.len() { verifier_state.prover_message::<[u8; 1]>().unwrap(); }

        let seed = verifier_state.verifier_message::<[u8; 32]>();

        let d = PolyRing::DEGREE;
        let mut c_zq_dev = DeviceVec::<Zq>::device_malloc(R * d).unwrap();
        let cfg = VecOpsConfig::default();
        icicle_core::random_sampling::random_sampling(true, &seed, &cfg, &mut c_zq_dev[..]).unwrap();

        let mut c_zq_host = vec![Zq::zero(); R * d];
        c_zq_dev.copy_to_host(HostSlice::from_mut_slice(&mut c_zq_host)).unwrap();
        let mut c_host = vec![PolyRing::zero(); R];
        for i in 0..R {
            let chunk = &c_zq_host[i * d..(i + 1) * d];
            c_host[i] = PolyRing::from_slice(chunk).unwrap();
        }
        DeviceVec::from_host_slice(&c_host)
    }
}

pub fn setup() -> PublicParams {
    // 1: A ← R_q^{n × δ₀·m}, note that in icicle matrices are one dimensional anyway only in matmult you can specify matrix dims
    let A_len = NROWS * DELTA0 * M;
    let A = DeviceVec::from_host_slice(&PolyRing::generate_random(A_len));
    // 2: B ← R_q^{n × n·δ·r}
    let B_len = NROWS * NROWS * DELTA * R;
    let B = DeviceVec::from_host_slice(&PolyRing::generate_random(B_len));
    // 3: D ← R_q^{n × δ·r}
    let D_len = NROWS * DELTA * R;
    let D = DeviceVec::from_host_slice(&PolyRing::generate_random(D_len));
    PublicParams { A, B, D }
}
// 2 - Commit function that will commit to the left side of the calculation a^T [s_1 | ... | s_r] and
//      also to vectors s and \tilde t.
pub struct CommitReturn {
    pub u: Vec<PolyRing>,
    pub s: Vec<Vec<PolyRing>>,
    pub t_hat_concat: Vec<PolyRing>,
}

pub fn commit(raw_f_coeffs: &[u64], pp: &PublicParams) -> Result<CommitReturn, IcicleError> {
    // checks: we need to check m * r is bigger or equal to N / d.
    assert!(M * R >= (N + PolyRing::DEGREE - 1) / PolyRing::DEGREE, "m*r must be >= N/d");
    // Line 1: define the function. Function coeffs are already defined with our input f_coeffs. Convert to Zq.

    //TODO: Note that coefficients of polynomial are assumed as u32 here. but looks good ow

    let f_coeffs: Vec<Zq> = raw_f_coeffs.iter().map(|&val| Zq::from(val as u32)).collect();
    let d = PolyRing::DEGREE;
    let cfg = VecOpsConfig::default();
    // Lines 2-3: chunk the coeffs in groups of d. Create chunks of size d and map them each to a PolyRing.
    // we add zeros to the extra parts at the end if there are any. Notice that at this stage we have the polynomial
	// coefficient matrix that is of size m x r. Each element in the matrix is a poly ring of degree d.


    let mut f_poly: Vec<PolyRing> = f_coeffs
        .chunks(d)
        .map(|chunk| PolyRing::from_slice(chunk).expect("chunk → PolyRing"))
        .collect();
    f_poly.resize(M * R, PolyRing::zero());

    // Line 4-5-6-7-8: for each of the r column vectors f_i^T, 1 through r, calculate their short version s_i, their commitment A*s_i,
	// and the short commitment t hat.	
	// notice that these data types are vectors that hold vectors of poly rings (each vector of poly ring is a column!)
    
    let mut s_vecs: Vec<Vec<PolyRing>> = Vec::with_capacity(R);
    // Total length of t̂ = r * (n * delta).
    let t_hat_total_len = R * NROWS * DELTA;
    let mut t_hat_concat_dev = DeviceVec::<PolyRing>::device_malloc(t_hat_total_len)?;

    let ntt_cfg = negacyclic_ntt::NegacyclicNttConfig::default();
    
    // NTT transform A once for all s_i multiplications
    // TODO: Ask whether I am doing NTT properly. This ntt transform SHOULD be faster than just matmul?

    let mut A_ntt = DeviceVec::<PolyRing>::device_malloc(pp.A.len())?;
    negacyclic_ntt::ntt(&pp.A, NTTDir::kForward, &ntt_cfg, &mut A_ntt[..])?;

    let start_commit_loop = Instant::now();
    for i in 1..(R + 1) {
        let f_i: Vec<PolyRing> = f_poly[(i-1)*M..(i)*M].to_vec();
        let f_i_dev = DeviceVec::from_host_slice(&f_i);

        let mut s_i = DeviceVec::<PolyRing>::device_malloc(M * DELTA0)?;
        balanced_decomposition::decompose::<PolyRing>(
            &f_i_dev[..],
            &mut s_i[..],
            (1 << B0) as u32,
            &cfg
        )?;

        // Move s_i to NTT domain for fast matmul
        let mut s_i_ntt = DeviceVec::<PolyRing>::device_malloc(s_i.len())?;
        negacyclic_ntt::ntt(&s_i, NTTDir::kForward, &ntt_cfg, &mut s_i_ntt[..])?;

        // t_i = A * s_i (in NTT domain)
        let mut t_i_ntt = DeviceVec::<PolyRing>::device_malloc(NROWS)?;
        matrix_ops::matmul::<PolyRing>(
            &A_ntt, NROWS as u32, A_COLS as u32,
            &s_i_ntt, A_COLS as u32, 1,
            &MatMulConfig::default(),
            &mut t_i_ntt,
        )?;

        // Inverse NTT to get t_i coefficients for next decomposition
        negacyclic_ntt::ntt_inplace(&mut t_i_ntt, NTTDir::kInverse, &ntt_cfg)?;

        let t_hat_i_len = NROWS * DELTA;
        let t_hat_i_start = (i - 1) * t_hat_i_len;
        
        // Decompose t_i -> t_hat_i directly on device
        balanced_decomposition::decompose::<PolyRing>(
            &t_i_ntt[..],
            &mut t_hat_concat_dev[t_hat_i_start..t_hat_i_start + t_hat_i_len],
            (1 << B1) as u32,
            &cfg,
        )?;

        // Collect s_i (host) for the proof
        let mut s_i_host = vec![PolyRing::zero(); s_i.len()];
        s_i.copy_to_host(HostSlice::from_mut_slice(&mut s_i_host))?;
        s_vecs.push(s_i_host);
    }
    println!("  [TIMER] Commit loop for R={} columns took: {:?}", R, start_commit_loop.elapsed());

    // NTT transform B and t_hat_concat for the final commitment u
    let start_commit_final = Instant::now();
    let mut B_ntt = DeviceVec::<PolyRing>::device_malloc(pp.B.len())?;
    negacyclic_ntt::ntt(&pp.B, NTTDir::kForward, &ntt_cfg, &mut B_ntt[..])?;

    let mut t_hat_concat_ntt = DeviceVec::<PolyRing>::device_malloc(t_hat_concat_dev.len())?;
    negacyclic_ntt::ntt(&t_hat_concat_dev, NTTDir::kForward, &ntt_cfg, &mut t_hat_concat_ntt[..])?;

    let mut u_ntt = DeviceVec::<PolyRing>::device_malloc(NROWS)?;
    matrix_ops::matmul::<PolyRing>(
        &B_ntt, NROWS as u32, B_COLS as u32,
        &t_hat_concat_ntt, B_COLS as u32, 1,
        &MatMulConfig::default(),
        &mut u_ntt,
    )?;

    negacyclic_ntt::ntt_inplace(&mut u_ntt, NTTDir::kInverse, &ntt_cfg)?;

    let mut u_host = vec![PolyRing::zero(); NROWS];
    u_ntt.copy_to_host(HostSlice::from_mut_slice(&mut u_host))?;

    let mut t_hat_concat_host = vec![PolyRing::zero(); t_hat_total_len];
    t_hat_concat_dev.copy_to_host(HostSlice::from_mut_slice(&mut t_hat_concat_host))?;
    println!("  [TIMER] Commit final NTTs & Matmul took: {:?}", start_commit_final.elapsed());

    Ok(CommitReturn {
        u: u_host,
        s: s_vecs,
        t_hat_concat: t_hat_concat_host,
    })
}
// Eval function for prover. We will simulate the verifier challenges using fiat shamir transform. 
// to do this, I will use https://github.com/arkworks-rs/spongefish. I will feed into the sha3 hash function these:
// the name of the protocol, the statement, commitment, every message sent and received including the challenges.
// obviously the secret should not be in the hash. After getting these, I will just continue on with the eval.
// Note that this function still needs 1 interaction from V, that is, the value of x, the eval point
// of the polynomial.

// -------------------------------------------------------------------------------------------------
// PROVER EVALUATION
// -------------------------------------------------------------------------------------------------

pub struct ProverProof {
    pub y: PolyRing,
    pub v: Vec<PolyRing>,
    // The Fiat-Shamir proof transcript sequence of bytes
    pub narg_transcript: Vec<u8>,
    // Labrador proof and transcript
    pub labrador_proof: labrador::LabradorProof,
    pub labrador_transcript: Vec<u8>,
}

/// Evaluate a polynomial f (given as Zq coefficients of length N) at a point x ∈ Zq.
/// Uses Horner's method: y = f[n-1], then y = y*x + f[n-2], ..., y = y*x + f[0].
// TODO: IS THERE A BETTER WAY ON GPU FOR THIS? I FEEL LIKE TRANSFERRING THE INSANE AMOUNT OF COEFFS ALONE
// WOULD BULLDOZE ANY EFFICIENCY GAIN ON GPU.
fn evaluate_polynomial(f_coeffs: &[Zq], x: &Zq) -> Zq {
    let mut y = Zq::zero();
    for i in (0..f_coeffs.len()).rev() {
        y = y * *x + f_coeffs[i];
    }
    y
}

/// Extract the constant term of a PolyRing element as a Zq scalar.
fn constant_term(y: &PolyRing) -> Zq {
    let bytes = unsafe {
        std::slice::from_raw_parts(y as *const _ as *const Zq, PolyRing::DEGREE)
    };
    bytes[0]
}

pub fn eval_prover(
    pp: &PublicParams,
    raw_f_coeffs: &[u64],
    commit_return: &CommitReturn,
    x_scalar: &Zq, //the value of eval point x
    lab_crs: &labrador::LabradorCRS,
) -> Result<ProverProof, IcicleError> {
    // convert raw f coeffs into zq elements
    let f_coeffs: Vec<Zq> = raw_f_coeffs.iter().map(|&val| Zq::from(val as u32)).collect();
    let d = PolyRing::DEGREE;
    let cfg = VecOpsConfig::default();
    let matcfg = MatMulConfig::default();

    // Lines 1-5
    
    // Construct powers of x: [1, x, x^2, ..., x^{d-1}]
    let mut x_powers = vec![Zq::zero(); d];
    let mut curr_x = Zq::one();
    for j in 0..d {
        x_powers[j] = curr_x;
        curr_x = curr_x * *x_scalar;
    }
    let x_d = curr_x; // We will use x^d for the scalar multiplication later

    // 2. Apply the Galois Automorphism σ_{-1}(x)
    // This maps coefficients [a_0, a_1, ..., a_{d-1}] to [a_0, -a_{d-1}, -a_{d-2}, ..., -a_1]
    // Please see my interim report for the mathematics explanation. But the point is that automorphism on negacyclic
    // rings such as the one we are using can be efficiently calculated using this formula.
    let mut sigma_x_coeffs = vec![Zq::zero(); d];
    sigma_x_coeffs[0] = x_powers[0];
    for j in 1..d {
        // Subtract from zero to negate in Zq
        sigma_x_coeffs[j] = Zq::zero() - x_powers[d - j]; 
    }
    // this is the sigma_{-1} in line 5
    let sigma_inv_x = PolyRing::from_slice(&sigma_x_coeffs).unwrap();

    // Line 5: Compute y = Σ σ_{-1}(x) * f_i * (x^d)^i
    let mut y_ring = PolyRing::zero();
    let mut curr_xd = Zq::one(); // Tracks (x^d)^i
    let num_chunks = N / d; // Number of polynomial chunks
    // read the "NOTE:" in helpers section for mul_poly add_poly use
    let start_y_eval = Instant::now();
    for i in 0..num_chunks {
        // Extract f_i chunk
        let chunk = &f_coeffs[i * d .. (i + 1) * d];
        let f_i = PolyRing::from_slice(chunk).unwrap();

        // Multiply polynomials: σ_{-1}(x) * f_i
        let term = mul_poly(&sigma_inv_x, &f_i);

        // Scale the resulting polynomial by the scalar (x^d)^i
        let term_coeffs = unsafe { std::slice::from_raw_parts(&term as *const _ as *const Zq, d) };
        let mut scaled_coeffs = vec![Zq::zero(); d];
        for k in 0..d {
            scaled_coeffs[k] = term_coeffs[k] * curr_xd;
        }
        let scaled_term = PolyRing::from_slice(&scaled_coeffs).unwrap();

        // Add to the running sum for y
        y_ring = add_poly(&y_ring, &scaled_term);

        // Update (x^d)^i for the next loop iteration
        curr_xd = curr_xd * x_d;
    }
    println!("  [TIMER] Prover evaluation of y_val took: {:?}", start_y_eval.elapsed());

    // Line 6. Note that the below loop simulates the gadget matrix multiplication. This is because only a part of the Kronecker
    // product matrix is relevant in G_{b0,m}. Specifically, we dot product 1 with the g matrix, then x^d, then x^2d ...
    // Another question that might arise is why coeffs is a zq vector instead of a single element zq. The answer is that
    // we are still working with polynomial rings, and even though every other element except the first one is zero, 
    // we still need the other elements to be there and be zero in order to have a ring.
    let x_d = {
        let mut xd = Zq::one();
        for _ in 0..d { xd = xd * *x_scalar; }
        xd
    };
    let mut a_host = vec![PolyRing::zero(); DELTA0 * M];
    let mut x_jd = Zq::one(); // x^{j*d}
    let start_a_host = Instant::now();
    for j in 0..M {
        let mut b0_pow = Zq::one(); // b0^k
        for k in 0..DELTA0 {
            let mut coeffs = vec![Zq::zero(); d];
            coeffs[0] = x_jd * b0_pow;
            a_host[k * M + j] = PolyRing::from_slice(&coeffs).unwrap();
            b0_pow = b0_pow * Zq::from((1 << B0) as u32);
        }
        x_jd = x_jd * x_d;
    }
    println!("  [TIMER] Prover computing a_host took: {:?}", start_a_host.elapsed());

    // Line 7: b^T
    let x_md = {
        let mut xmd = Zq::one();
        for _ in 0..(M * d) { xmd = xmd * *x_scalar; }
        xmd
    };
    let mut b_host = vec![PolyRing::zero(); R];
    let mut x_imd = Zq::one(); // x^{i*m*d}
    for i in 0..R {
        let mut coeffs = vec![Zq::zero(); d];
        coeffs[0] = x_imd;
        b_host[i] = PolyRing::from_slice(&coeffs).unwrap();
        x_imd = x_imd * x_md;
    }

    // Line 8: w^T = a^T [s_1 | ... | s_r]
    // w[i] = ⟨a, s_i⟩
    let mut a_dev = DeviceVec::from_host_slice(&a_host);
    let ntt_cfg = negacyclic_ntt::NegacyclicNttConfig::default();
    negacyclic_ntt::ntt_inplace(&mut a_dev, NTTDir::kForward, &ntt_cfg)?;
    
    let mut w_host = Vec::with_capacity(R);
    let mut tmp_dot = DeviceVec::<PolyRing>::device_malloc(1)?;
    let mut s_i_dev = DeviceVec::<PolyRing>::device_malloc(DELTA0 * M)?;
    println!("  [DEBUG] Starting w-reduction loop, R={}", R);
    let start_w_reduction = Instant::now();
    for i in 0..R {
        if i % 50 == 0 { println!("  [DEBUG] w-reduction iteration {}", i); }
        s_i_dev.copy_from_host(HostSlice::from_slice(&commit_return.s[i]))?;
        negacyclic_ntt::ntt_inplace(&mut s_i_dev, NTTDir::kForward, &ntt_cfg)?;
        
        matrix_ops::matmul::<PolyRing>(
            &a_dev, 1, (DELTA0 * M) as u32,        // a^T is 1 x K
            &s_i_dev, (DELTA0 * M) as u32, 1,      // s_i is K x 1
            &MatMulConfig::default(),
            &mut tmp_dot,
        )?;
        
        negacyclic_ntt::ntt_inplace(&mut tmp_dot, NTTDir::kInverse, &ntt_cfg)?;
        let mut res = [PolyRing::zero()];
        tmp_dot.copy_to_host(HostSlice::from_mut_slice(&mut res))?;
        w_host.push(res[0]);
    }
    println!("  [DEBUG] w-reduction loop finished.");
    println!("  [TIMER] Prover w-reduction took: {:?}", start_w_reduction.elapsed());
    let w_dev = DeviceVec::from_host_slice(&w_host);

    // Line 9: w_hat = G^{-1}_{b,r}(w)
    let w_hat_len = DELTA * R;
    let mut w_hat = DeviceVec::<PolyRing>::device_malloc(w_hat_len)?;
    balanced_decomposition::decompose::<PolyRing>(
        &w_dev[..], &mut w_hat[..], (1 << B1) as u32, &cfg,
    )?;

    // Line 10: v = D · w_hat
    let mut v_dev = DeviceVec::<PolyRing>::device_malloc(NROWS)?;
    matrix_ops::matmul::<PolyRing>(
        &pp.D, NROWS as u32, D_COLS as u32,
        &w_hat, D_COLS as u32, 1,
        &matcfg, &mut v_dev,
    )?;
    let mut v_host = vec![PolyRing::zero(); NROWS];
    v_dev.copy_to_host(HostSlice::from_mut_slice(&mut v_host))?;

    let mut w_hat_host = vec![PolyRing::zero(); w_hat_len];
    w_hat.copy_to_host(HostSlice::from_mut_slice(&mut w_hat_host))?;

    // FIAT SHAMIR CHALLENGE (line 12)
    let domain_sep = DomainSeparator::new(FiatShamirGreyhound::protocol_id())
        .session(spongefish::session!("greyhound_pcs"))
        .instance(&[0u8; 0]);
    let mut prover_state = domain_sep.std_prover();
    let c = FiatShamirGreyhound::prove(&mut prover_state, &commit_return.u, x_scalar, &y_ring, &v_dev);
    let narg_transcript = prover_state.narg_string().to_vec();

    // Copy challenge c to host
    let mut c_host = vec![PolyRing::zero(); R];
    c.copy_to_host(HostSlice::from_mut_slice(&mut c_host))?;

    // Lines 14 - 17
    // Build Labrador witness from the composited z_full blocks
    let z_len = DELTA0 * M;
    let w_len = DELTA;
    let t_len = NROWS * DELTA;
    let mut z_full: Vec<Vec<PolyRing>> = Vec::with_capacity(R);
    let start_z_full_pack = Instant::now();
    for i in 0..R {
        let mut block = Vec::with_capacity(w_len + t_len + z_len);
        
        // Gather the interleaved digits for the witness block
        let mut w_hat_i = vec![PolyRing::zero(); DELTA];
        for k in 0..DELTA {
            w_hat_i[k] = w_hat_host[k * R + i];
        }
        block.extend_from_slice(&w_hat_i);
        
        let t_start = i * t_len;
        block.extend_from_slice(&commit_return.t_hat_concat[t_start..t_start + t_len]);
        block.extend_from_slice(&commit_return.s[i]);
        z_full.push(block);
    }
    println!("  [TIMER] Prover packing z_full blocks took: {:?}", start_z_full_pack.elapsed());
    
    // Build the constraint system: P matrix components + target h
    let constraint = labrador::ConstraintSystem {
        a_eval: a_host.clone(),
        b_eval: b_host.clone(),
        sigma_inv_x: sigma_inv_x,
        A_ajtai: {
            let mut A_h = vec![PolyRing::zero(); pp.A.len()];
            pp.A.copy_to_host(HostSlice::from_mut_slice(&mut A_h))?;
            A_h
        },
        h_target: {
            let mut h = Vec::new();
            h.extend_from_slice(&v_host);         // v (NROWS)
            h.extend_from_slice(&commit_return.u); // u (NROWS)
            h.push(y_ring);                        // sigma^{-1}(x)^{-1} * y (1)
            h.push(PolyRing::zero());              // 0 (1) 
            h
        },
        n_w: w_len,
        n_t: t_len,
        n_s: z_len,
        n_commit: NROWS,
    };

    let lab_witness = labrador::LabradorWitness {
        s: z_full,
    };
    println!("  [DEBUG] Labrador witness finished.");

    // Run Labrador prover (using shared CRS)
    let (labrador_proof, labrador_transcript) = labrador::labrador_prove(lab_crs, &lab_witness, &constraint)?;
    println!("  [DEBUG] Labrador prove finished.");
    Ok(ProverProof {
        y: y_ring,
        v: v_host,
        narg_transcript,
        labrador_proof,
        labrador_transcript,
    })
}

// -------------------------------------------------------------------------------------------------
// VERIFIER EVALUATION
// -------------------------------------------------------------------------------------------------

pub fn eval_verifier(
    pp: &PublicParams,
    commit_return: &CommitReturn,
    x_scalar: &Zq,
    proof: &ProverProof,
    raw_f_coeffs: &[u64],
    lab_crs: &labrador::LabradorCRS,
) -> Result<bool, IcicleError> {
    let d = PolyRing::DEGREE;

    // 1. Recover challenge vector c via Fiat-Shamir
    let domain_sep = DomainSeparator::new(FiatShamirGreyhound::protocol_id())
        .session(spongefish::session!("greyhound_pcs"))
        .instance(&[0u8; 0]);
    let v_dev = DeviceVec::from_host_slice(&proof.v);
    let verifier_state = domain_sep.std_verifier(&proof.narg_transcript);
    let c = FiatShamirGreyhound::verify(verifier_state, &commit_return.u, x_scalar, &proof.y, &v_dev);
    let mut c_host = vec![PolyRing::zero(); R];
    c.copy_to_host(HostSlice::from_mut_slice(&mut c_host))?;

    // 2. Check ct(y) == f(x)
    let f_coeffs: Vec<Zq> = raw_f_coeffs.iter().map(|&val| Zq::from(val as u32)).collect();
    let ct_y = constant_term(&proof.y);
    let f_x = evaluate_polynomial(&f_coeffs, x_scalar);
    if !zq_equal(&ct_y, &f_x) {
        println!("  [FAIL] Check 1: ct(y) != f(x)");
        return Ok(false);
    }
    println!("  [PASS] Check 1: ct(y) == f(x)");

    // 3. Rebuild the constraint system from public info
    let x_d = {
        let mut xd = Zq::one();
        for _ in 0..d { xd = xd * *x_scalar; }
        xd
    };
    let mut a_host = vec![PolyRing::zero(); DELTA0 * M];
    let mut x_jd = Zq::one();
    for j in 0..M {
        let mut b0_pow = Zq::one();
        for k in 0..DELTA0 {
            let mut coeffs = vec![Zq::zero(); d];
            coeffs[0] = x_jd * b0_pow;
            a_host[k * M + j] = PolyRing::from_slice(&coeffs).unwrap();
            b0_pow = b0_pow * Zq::from((1 << B0) as u32);
        }
        x_jd = x_jd * x_d;
    }
    let x_md = {
        let mut xmd = Zq::one();
        for _ in 0..(M * d) { xmd = xmd * *x_scalar; }
        xmd
    };
    let mut b_host = vec![PolyRing::zero(); R];
    let mut x_imd = Zq::one();
    for i in 0..R {
        let mut coeffs = vec![Zq::zero(); d];
        coeffs[0] = x_imd;
        b_host[i] = PolyRing::from_slice(&coeffs).unwrap();
        x_imd = x_imd * x_md;
    }
    let mut sigma_x_coeffs = vec![Zq::zero(); d];
    {
        let mut x_powers = vec![Zq::zero(); d];
        let mut curr_x = Zq::one();
        for j in 0..d {
            x_powers[j] = curr_x;
            curr_x = curr_x * *x_scalar;
        }
        sigma_x_coeffs[0] = x_powers[0];
        for j in 1..d {
            sigma_x_coeffs[j] = Zq::zero() - x_powers[d - j];
        }
    }
    let sigma_inv_x = PolyRing::from_slice(&sigma_x_coeffs).unwrap();

    let constraint = labrador::ConstraintSystem {
        a_eval: a_host,
        b_eval: b_host,
        sigma_inv_x,
        A_ajtai: {
            let mut A_h = vec![PolyRing::zero(); pp.A.len()];
            pp.A.copy_to_host(HostSlice::from_mut_slice(&mut A_h))?;
            A_h
        },
        h_target: {
            let mut h = Vec::new();
            h.extend_from_slice(&proof.v);          // v (NROWS)
            h.extend_from_slice(&commit_return.u);  // u (NROWS)
            h.push(proof.y);                         // sigma^{-1}(x)^{-1} * y (1)
            h.push(PolyRing::zero());                // 0 (1)
            h
        },
        n_w: DELTA,
        n_t: NROWS * DELTA,
        n_s: DELTA0 * M,
        n_commit: NROWS,
    };

    // 4. Run Labrador Verifier — all SIS checks are inside
    let lab_ok = labrador::labrador_verify(
        lab_crs,
        &proof.labrador_proof,
        &proof.labrador_transcript,
        &constraint,
    )?;
    if !lab_ok {
        println!("  [FAIL] Check 2: Labrador verification failed");
        return Ok(false);
    }
    println!("  [PASS] Check 2: Labrador verification passed");

    Ok(true)
}

// Helpers
// NOTE: notice that these functions are called ONLY ONCE on single polynomials. That is, polynomials
// of degree d. Which means we only have a single polyring of 64 coefficients. I figure that in this case it is much
// faster to just do the operations on cpu directly.
fn add_poly(a: &PolyRing, b: &PolyRing) -> PolyRing {
    let d = PolyRing::DEGREE;
    let a_bytes = unsafe { std::slice::from_raw_parts(a as *const _ as *const Zq, d) };
    let b_bytes = unsafe { std::slice::from_raw_parts(b as *const _ as *const Zq, d) };
    let mut sum_coeffs = vec![Zq::zero(); d];
    for i in 0..d {
        sum_coeffs[i] = a_bytes[i] + b_bytes[i];
    }
    PolyRing::from_slice(&sum_coeffs).unwrap()
}

fn mul_poly(a: &PolyRing, b: &PolyRing) -> PolyRing {
    let d = PolyRing::DEGREE;
    let a_bytes = unsafe { std::slice::from_raw_parts(a as *const _ as *const Zq, d) };
    let b_bytes = unsafe { std::slice::from_raw_parts(b as *const _ as *const Zq, d) };
    let mut prod = vec![Zq::zero(); d * 2];
    for i in 0..d {
        for j in 0..d {
            prod[i + j] = prod[i + j] + a_bytes[i] * b_bytes[j];
        }
    }
    let mut res_coeffs = vec![Zq::zero(); d];
    for i in 0..d {
        res_coeffs[i] = prod[i] - prod[i + d];
    }
    PolyRing::from_slice(&res_coeffs).unwrap()
}

fn zq_equal(a: &Zq, b: &Zq) -> bool {
    let size = std::mem::size_of::<Zq>();
    let a_bytes = unsafe { std::slice::from_raw_parts(a as *const _ as *const u8, size) };
    let b_bytes = unsafe { std::slice::from_raw_parts(b as *const _ as *const u8, size) };
    a_bytes == b_bytes
}

fn main() {
    println!("Loading default backend and initializing device...");
    let _ = icicle_runtime::runtime::load_backend("/workspace/icicle").unwrap();
    let device = icicle_runtime::Device::new("CUDA", 0);
    // let device = icicle_runtime::Device::new("CUDA", 0);
    icicle_runtime::set_device(&device).unwrap();

    println!("Setting up Greyhound parameters: M={}, R={}", M, R);
    let pp = setup();

    // Calculate the theoretical squared norm bound (\bar{\gamma}^2) exactly as defined in Greyhound Step 16
    let d_f64 = PolyRing::DEGREE as f64;
    let b0_f64 = (1u64 << B0) as f64;
    let b1_f64 = (1u64 << B1) as f64;
    let kappa = 51.0; // Typical l1 norm bound for a Fiat-Shamir challenge polynomial

    let term1 = b1_f64.powi(2) * (NROWS as f64 + 1.0) * (DELTA as f64) * (R as f64) * d_f64;
    let term2 = ((R as f64) * kappa * b0_f64).powi(2) * (DELTA0 as f64) * (M as f64) * d_f64;
    let gamma_bar_sq = term1 + term2;

    println!("Setting up Labrador CRS with theoretical norm bound sq: {:.2e}", gamma_bar_sq);

    // Create shared Labrador CRS
    let lab_crs = labrador::LabradorCRS::setup(
        R, DELTA0 * M,
        NROWS, N1, N1,
        DELTA, DELTA,
        (1 << B1) as u32, (1 << B1) as u32, (1 << B0) as u32, // b, b1, b2
        1, 1, gamma_bar_sq,
    );

    let poly_size = N;
    let raw_f_coeffs: Vec<u64> = (0..poly_size).map(|i| (i % 100) as u64).collect();

    println!("Committing to function size {} elements...", poly_size);
    let commit_ret = commit(&raw_f_coeffs, &pp).expect("Commit failed");

    let mut rng = rand::thread_rng();
    let x_random_value : u32 = rng.r#gen::<u32>();
    let x_eval = Zq::from(x_random_value); 
    println!("Running prover evaluation at x= {} ...", x_random_value);

    let proof = eval_prover(&pp, &raw_f_coeffs, &commit_ret, &x_eval, &lab_crs).expect("Prover failed");
    
    println!("Running verifier evaluation...");
    let verified = eval_verifier(&pp, &commit_ret, &x_eval, &proof, &raw_f_coeffs, &lab_crs).expect("Verifier failed");

    if verified {
        println!("Verifier PASSED successfully!");
    } else {
        println!("Verifier REJECTED the proof!");
    }
}

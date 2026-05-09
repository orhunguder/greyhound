#![allow(non_snake_case)]
use icicle_core::{
    matrix_ops::{self, MatMulConfig},
    vec_ops::{VecOpsConfig},
    traits::TryInverse,
    balanced_decomposition,
    negacyclic_ntt,
    ntt::NTTDir,
    polynomial_ring::{flatten_polyring_slice_mut, PolynomialRing},
    random_sampling,
    random_sampling::challenge_space_polynomials_sampling,
    bignum::BigNum,
};
mod labrador;
use icicle_babykoala::{
    ring::ScalarRing as Zq,
    polynomial_ring::PolyRing,
};
use icicle_runtime::{
    errors::eIcicleError,
    memory::{DeviceVec, HostSlice, HostOrDeviceSlice},
    IcicleError,
};
use spongefish::{
    protocol_id, DomainSeparator, ProverState, VerifierState,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
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
    //TODO!!!! LABRADOR PARAMS ARE GIVEN TO LABRADOR BUT ARE NEVER COMMITED TO!!!!
    //MIGHT ENABLE SOME ATTACK
    // pub pp_prime //Placeholder for labrador params
}

// this is the struct for the fiat shamir.  Must include a prove and verify function.
struct FiatShamirGreyhound;

#[derive(Clone, Copy)]
struct GreyhoundTranscriptParams {
    n: u32,
    m: u32,
    r: u32,
    nrows: u32,
    b0: u32,
    delta0: u32,
    b1: u32,
    delta: u32,
}

fn greyhound_transcript_params() -> GreyhoundTranscriptParams {
    GreyhoundTranscriptParams {
        n: N as u32,
        m: M as u32,
        r: R as u32,
        nrows: NROWS as u32,
        b0: B0 as u32,
        delta0: DELTA0 as u32,
        b1: B1 as u32,
        delta: DELTA as u32,
    }
}

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

// fn to_bytes_host<T: Copy>(host_vec: &Vec<T>) -> Vec<u8> {
//     let len = host_vec.len();
//     if len == 0 {
//         return vec![];
//     }
//     let byte_len = len * std::mem::size_of::<T>();
//     let bytes = unsafe { std::slice::from_raw_parts(host_vec.as_ptr() as *const u8, byte_len) };
//     bytes.to_vec()
// }

fn zq_to_bytes(scalar: &Zq) -> Vec<u8> {
    let byte_len = std::mem::size_of::<Zq>();
    let bytes = unsafe { std::slice::from_raw_parts(scalar as *const _ as *const u8, byte_len) };
    bytes.to_vec()
}

fn poly_to_bytes(y_ring: &PolyRing) -> Vec<u8> {
    let bytes = unsafe { std::slice::from_raw_parts(y_ring as *const _ as *const u8, std::mem::size_of::<PolyRing>()) };
    bytes.to_vec()
}

fn transcript_params_to_bytes(params: &GreyhoundTranscriptParams) -> Vec<u8> {
    let words = [
        params.n,
        params.m,
        params.r,
        params.nrows,
        params.b0,
        params.delta0,
        params.b1,
        params.delta,
    ];
    let bytes = unsafe {
        std::slice::from_raw_parts(
            words.as_ptr() as *const u8,
            words.len() * std::mem::size_of::<u32>(),
        )
    };
    bytes.to_vec()
}

fn sample_greyhound_challenges(seed: &[u8; 32]) -> DeviceVec<PolyRing> {
    let mut c_dev = DeviceVec::<PolyRing>::device_malloc(R).unwrap();
    challenge_space_polynomials_sampling(
        seed,
        &VecOpsConfig::default(),
        31, // ±1 coefficients
        10, // ±2 coefficients
        15, // operator norm bound
        &mut c_dev,
    )
    .unwrap();
    c_dev
}

impl FiatShamirGreyhound {
    pub fn protocol_id() -> [u8; 64] {
        protocol_id(core::format_args!("greyhound proof"))
    }

    pub fn prove(
        prover_state: &mut ProverState,
        u: &DeviceVec<PolyRing>,    // Public Commitment
        x: &Zq,                     // Evaluation point
        y: &PolyRing,               // Evaluation result
        v: &DeviceVec<PolyRing>,    // Prover's message v
    ) -> DeviceVec<PolyRing> {
        Self::prove_with_params(prover_state, greyhound_transcript_params(), u, x, y, v)
    }

    fn prove_with_params(
        prover_state: &mut ProverState,
        params: GreyhoundTranscriptParams,
        u: &DeviceVec<PolyRing>,
        x: &Zq,
        y: &PolyRing,
        v: &DeviceVec<PolyRing>,
    ) -> DeviceVec<PolyRing> {
        // Absorb Prover Messages
        for &b in &transcript_params_to_bytes(&params) {
            prover_state.prover_message(&[b]);
        }
        for &b in &to_bytes(u) {
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

        // Squeeze challenge
        let seed = prover_state.verifier_message::<[u8; 32]>();

        sample_greyhound_challenges(&seed)
    }

    // Evaluates the verifier side using the sponge state.
    pub fn verify(
        verifier_state: VerifierState,
        u: &DeviceVec<PolyRing>, 
        x: &Zq, 
        y: &PolyRing, 
        v: &DeviceVec<PolyRing>
    ) -> DeviceVec<PolyRing> {
        Self::verify_with_params(verifier_state, greyhound_transcript_params(), u, x, y, v)
    }

    fn verify_with_params(
        mut verifier_state: VerifierState,
        params: GreyhoundTranscriptParams,
        u: &DeviceVec<PolyRing>,
        x: &Zq,
        y: &PolyRing,
        v: &DeviceVec<PolyRing>,
    ) -> DeviceVec<PolyRing> {
        // TODO: IDK about u8?
        let pb = transcript_params_to_bytes(&params);
        for _ in 0..pb.len() { verifier_state.prover_message::<[u8; 1]>().unwrap(); }
        let ub = to_bytes(u);
        for _ in 0..ub.len() { verifier_state.prover_message::<[u8; 1]>().unwrap(); }
        let xb = zq_to_bytes(x);
        for _ in 0..xb.len() { verifier_state.prover_message::<[u8; 1]>().unwrap(); }
        let yb = poly_to_bytes(y);
        for _ in 0..yb.len() { verifier_state.prover_message::<[u8; 1]>().unwrap(); }
        let vb = to_bytes(v);
        for _ in 0..vb.len() { verifier_state.prover_message::<[u8; 1]>().unwrap(); }

        let seed = verifier_state.verifier_message::<[u8; 32]>();

        sample_greyhound_challenges(&seed)
    }
}

pub fn setup() -> PublicParams {
    fn sample_device_polyrings(len: usize, seed: [u8; 32]) -> DeviceVec<PolyRing> {
        let mut out = DeviceVec::<PolyRing>::device_malloc(len).expect("CRS device allocation failed");
        {
            let mut out_coeffs = flatten_polyring_slice_mut(&mut out);
            random_sampling::random_sampling(true, &seed, &VecOpsConfig::default(), &mut out_coeffs)
                .expect("CRS device random sampling failed");
        }
        out
    }

    // 1: A ← R_q^{n × δ₀·m}, note that in icicle matrices are one dimensional anyway only in matmult you can specify matrix dims
    let A_len = NROWS * DELTA0 * M;
    let A = sample_device_polyrings(A_len, [0xA1; 32]);
    // 2: B ← R_q^{n × n·δ·r}
    let B_len = NROWS * NROWS * DELTA * R;
    let B = sample_device_polyrings(B_len, [0xB2; 32]);
    // 3: D ← R_q^{n × δ·r}
    let D_len = NROWS * DELTA * R;
    let D = sample_device_polyrings(D_len, [0xD3; 32]);
    PublicParams { A, B, D }
}
// 2 - Commit function that will commit to the left side of the calculation a^T [s_1 | ... | s_r] and
//      also to vectors s and \tilde t.
pub struct CommitReturn {
    pub u: DeviceVec<PolyRing>,
    pub s: DeviceVec<PolyRing>,
    pub t_hat_concat: DeviceVec<PolyRing>,
}

pub fn commit(raw_f_coeffs: &[u32], pp: &PublicParams) -> Result<CommitReturn, IcicleError> {
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
    drop(f_coeffs);
    f_poly.resize(M * R, PolyRing::zero());

    // Line 4-5-6-7-8: for each of the r column vectors f_i^T, 1 through r, calculate their short version s_i, their commitment A*s_i,
	// and the short commitment t hat.	
	// notice that these data types are vectors that hold vectors of poly rings (each vector of poly ring is a column!)
    
    // Total length of t̂ = r * (n * delta).
    let t_hat_total_len = R * NROWS * DELTA;

    let ntt_cfg = negacyclic_ntt::NegacyclicNttConfig::default();
    
    let _start_commit_batched = Instant::now();
    let target_size = M * R;
    let f_poly_dev = DeviceVec::from_host_slice(&f_poly);
    drop(f_poly);

    let mut s_decomp_dev = DeviceVec::<PolyRing>::device_malloc(target_size * DELTA0)?;
    balanced_decomposition::decompose::<PolyRing>(
        &f_poly_dev[..],
        &mut s_decomp_dev[..],
        (1 << B0) as u32,
        &cfg,
    )?;

    let mut s_decomp_global = vec![PolyRing::zero(); target_size * DELTA0];
    s_decomp_dev.copy_to_host(HostSlice::from_mut_slice(&mut s_decomp_global))?;

    let mut s_flat_host = vec![PolyRing::zero(); R * A_COLS];
    for i in 0..R {
        for k in 0..DELTA0 {
            for j in 0..M {
                let src = k * target_size + i * M + j;
                let dst = i * A_COLS + k * M + j;
                s_flat_host[dst] = s_decomp_global[src];
            }
        }
    }
    drop(s_decomp_global);

    let s_flat_dev = DeviceVec::from_host_slice(&s_flat_host);

    let mut A_ntt = DeviceVec::<PolyRing>::device_malloc(pp.A.len())?;
    negacyclic_ntt::ntt(&pp.A, NTTDir::kForward, &ntt_cfg, &mut A_ntt[..])?;

    let mut s_flat_ntt_dev = DeviceVec::<PolyRing>::device_malloc(s_flat_dev.len())?;
    negacyclic_ntt::ntt(&s_flat_dev, NTTDir::kForward, &ntt_cfg, &mut s_flat_ntt_dev[..])?;
    drop(s_flat_host);

    let t_total_len = R * NROWS;
    let mut t_rows_by_r_ntt_dev = DeviceVec::<PolyRing>::device_malloc(t_total_len)?;
    let mut matcfg_batch = MatMulConfig::default();
    matcfg_batch.b_transposed = true;
    matrix_ops::matmul::<PolyRing>(
        &A_ntt, NROWS as u32, A_COLS as u32,
        &s_flat_ntt_dev, R as u32, A_COLS as u32,
        &matcfg_batch,
        &mut t_rows_by_r_ntt_dev,
    )?;

    negacyclic_ntt::ntt_inplace(&mut t_rows_by_r_ntt_dev, NTTDir::kInverse, &ntt_cfg)?;

    let mut t_rows_by_r_host = vec![PolyRing::zero(); t_total_len];
    t_rows_by_r_ntt_dev.copy_to_host(HostSlice::from_mut_slice(&mut t_rows_by_r_host))?;

    let mut t_by_i_host = vec![PolyRing::zero(); t_total_len];
    for row in 0..NROWS {
        for i in 0..R {
            t_by_i_host[i * NROWS + row] = t_rows_by_r_host[row * R + i];
        }
    }
    drop(t_rows_by_r_host);

    let t_by_i_dev = DeviceVec::from_host_slice(&t_by_i_host);
    drop(t_by_i_host);
    let mut t_decomp_dev = DeviceVec::<PolyRing>::device_malloc(t_total_len * DELTA)?;
    balanced_decomposition::decompose::<PolyRing>(
        &t_by_i_dev[..],
        &mut t_decomp_dev[..],
        (1 << B1) as u32,
        &cfg,
    )?;

    let mut t_decomp_global = vec![PolyRing::zero(); t_total_len * DELTA];
    t_decomp_dev.copy_to_host(HostSlice::from_mut_slice(&mut t_decomp_global))?;

    let mut t_hat_concat_host = vec![PolyRing::zero(); t_hat_total_len];
    for i in 0..R {
        for k in 0..DELTA {
            for row in 0..NROWS {
                let src = k * t_total_len + i * NROWS + row;
                let dst = i * (NROWS * DELTA) + k * NROWS + row;
                t_hat_concat_host[dst] = t_decomp_global[src];
            }
        }
    }
    drop(t_decomp_global);
    // println!("  [TIMER] Batched commit decomposition and A*S took: {:?}", start_commit_batched.elapsed());

    // NTT transform B and t_hat_concat for the final commitment u
    let _start_commit_final = Instant::now();
    let mut B_ntt = DeviceVec::<PolyRing>::device_malloc(pp.B.len())?;
    negacyclic_ntt::ntt(&pp.B, NTTDir::kForward, &ntt_cfg, &mut B_ntt[..])?;

    let t_hat_concat_dev = DeviceVec::from_host_slice(&t_hat_concat_host);
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

    // println!("  [TIMER] Commit final NTTs & Matmul took: {:?}", start_commit_final.elapsed());

    Ok(CommitReturn {
        u: u_ntt,
        s: s_flat_dev,
        t_hat_concat: t_hat_concat_dev,
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

#[derive(Clone)]
pub struct ProverProof {
    pub y: PolyRing,
    pub claimed_eval: Zq,
    pub v: Vec<PolyRing>,
    // The Fiat-Shamir proof transcript sequence of bytes
    pub narg_transcript: Vec<u8>,
    // Labrador proof and transcript
    pub labrador_proof: labrador::LabradorProof,
    pub labrador_transcript: Vec<u8>,
    pub labrador_prove_elapsed: std::time::Duration,
}

fn sigma_inverse_apply(y: &PolyRing, x_scalar: &Zq) -> PolyRing {
    let d = PolyRing::DEGREE;
    let mut x_pow = Zq::one();
    for _ in 0..d {
        x_pow = x_pow * *x_scalar;
    }

    let denom_inv = (Zq::one() + x_pow)
        .try_inv()
        .expect("sigma inverse denominator is zero");
    let mut inverse_coeffs = vec![Zq::zero(); d];
    inverse_coeffs[0] = denom_inv;
    inverse_coeffs[d - 1] = *x_scalar * denom_inv;

    mul_poly(&PolyRing::from_slice(&inverse_coeffs).unwrap(), y)
}

// OLD VALIDATION HELPER: kept for re-enabling old-vs-factored y checks.
// fn poly_equal_host(a: &PolyRing, b: &PolyRing) -> bool {
//     let size = std::mem::size_of::<PolyRing>();
//     let a_bytes = unsafe { std::slice::from_raw_parts(a as *const _ as *const u8, size) };
//     let b_bytes = unsafe { std::slice::from_raw_parts(b as *const _ as *const u8, size) };
//     a_bytes == b_bytes
// }

pub fn eval_prover(
    pp: &PublicParams,
    raw_f_coeffs: &[u32],
    commit_return: &CommitReturn,
    x_scalar: &Zq, //the value of eval point x
    lab_crs: &labrador::LabradorCRS,
) -> Result<ProverProof, IcicleError> {
    // convert raw f coeffs into zq elements
    let f_coeffs: Vec<Zq> = raw_f_coeffs.iter().map(|&val| Zq::from(val as u32)).collect();
    let d = PolyRing::DEGREE;
    let cfg = VecOpsConfig::default();
    let claimed_eval = evaluate_polynomial(raw_f_coeffs, x_scalar);

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

    // Line 5: Compute y = σ_{-1}(x) * (Σ f_i * (x^d)^i).
    // OLD VALIDATED PATH:
    // let mut y_ring_old = PolyRing::zero();
    // let mut sigma_inv_y_target_old = PolyRing::zero();
    // let mut curr_xd_old = Zq::one();
    // for i in 0..num_chunks {
    //     let chunk = &f_coeffs[i * d .. (i + 1) * d];
    //     let f_i = PolyRing::from_slice(chunk).unwrap();
    //     let term = mul_poly(&sigma_inv_x, &f_i);
    //     let term_coeffs = unsafe { std::slice::from_raw_parts(&term as *const _ as *const Zq, d) };
    //     let f_i_coeffs = unsafe { std::slice::from_raw_parts(&f_i as *const _ as *const Zq, d) };
    //     let mut scaled_coeffs = vec![Zq::zero(); d];
    //     let mut preimage_coeffs = vec![Zq::zero(); d];
    //     for k in 0..d {
    //         scaled_coeffs[k] = term_coeffs[k] * curr_xd_old;
    //         preimage_coeffs[k] = f_i_coeffs[k] * curr_xd_old;
    //     }
    //     let scaled_term = PolyRing::from_slice(&scaled_coeffs).unwrap();
    //     let preimage_term = PolyRing::from_slice(&preimage_coeffs).unwrap();
    //     y_ring_old = add_poly(&y_ring_old, &scaled_term);
    //     sigma_inv_y_target_old = add_poly(&sigma_inv_y_target_old, &preimage_term);
    //     curr_xd_old = curr_xd_old * x_d;
    // }
    // let y_ring_fast = mul_poly(&sigma_inv_x, &sigma_inv_y_target_old);
    // assert!(poly_equal_host(&y_ring_old, &y_ring_fast));

    let mut sigma_inv_y_target = PolyRing::zero();
    let mut curr_xd = Zq::one(); // Tracks (x^d)^i
    let num_chunks = N / d; // Number of polynomial chunks
    let _start_y_eval = Instant::now();
    for i in 0..num_chunks {
        let chunk = &f_coeffs[i * d .. (i + 1) * d];
        let f_i = PolyRing::from_slice(chunk).unwrap();
        let f_i_coeffs = unsafe { std::slice::from_raw_parts(&f_i as *const _ as *const Zq, d) };
        let mut preimage_coeffs = vec![Zq::zero(); d];
        for k in 0..d {
            preimage_coeffs[k] = f_i_coeffs[k] * curr_xd;
        }
        let preimage_term = PolyRing::from_slice(&preimage_coeffs).unwrap();
        sigma_inv_y_target = add_poly(&sigma_inv_y_target, &preimage_term);
        curr_xd = curr_xd * x_d;
    }
    let y_ring = mul_poly(&sigma_inv_x, &sigma_inv_y_target);
    // println!("  [TIMER] Prover evaluation of y_val factored path took: {:?}", start_y_eval.elapsed());
    let proof_eval = constant_term(&y_ring);
    if !zq_equal(&proof_eval, &claimed_eval) {
        // println!("  [FAIL] Prover claim check: ct(y) != claimed f(x)");
        return Err(IcicleError::new(
            eIcicleError::InvalidArgument,
            "prover computed y does not match claimed evaluation",
        ));
    }
    // println!("  [PASS] Prover claim check: ct(y) matches claimed f(x)");

    // Line 6. Note that the below loop simulates the gadget matrix multiplication. This is because only a part of the Kronecker
    // product matrix is relevant in G_{b0,m}. Specifically, we dot product 1 with the g matrix, then x^d, then x^2d ...
    // Another question that might arise is why coeffs is a zq vector instead of a single element zq. The answer is that
    // we are still working with polynomial rings, and even though every other element except the first one is zero, 
    // we still need the other elements to be there and be zero in order to have a ring for the multiplication
    // in line 8.
    // I always get confused here but the math is correct you can verify for yourself keeping in mind that
    // G_{b0,m} is of shape m x (m * delta_zero)
    let x_d = {
        let mut xd = Zq::one();
        for _ in 0..d { xd = xd * *x_scalar; }
        xd
    };
    let mut a_host = vec![PolyRing::zero(); DELTA0 * M];
    let mut x_jd = Zq::one(); // x^{j*d}
    let _start_a_host = Instant::now();
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
    // println!("  [TIMER] Prover computing a_host took: {:?}", start_a_host.elapsed());

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
    let s_len = DELTA0 * M;
    let mut a_dev = DeviceVec::from_host_slice(&a_host);
    let ntt_cfg = negacyclic_ntt::NegacyclicNttConfig::default();
    negacyclic_ntt::ntt_inplace(&mut a_dev, NTTDir::kForward, &ntt_cfg)?;

    assert_eq!(commit_return.s.len(), R * s_len, "commit_return.s has invalid flat size");
    let mut s_flat_ntt_dev = DeviceVec::<PolyRing>::device_malloc(commit_return.s.len())?;
    negacyclic_ntt::ntt(&commit_return.s, NTTDir::kForward, &ntt_cfg, &mut s_flat_ntt_dev[..])?;

    // println!("  [DEBUG] Starting batched w-reduction, R={}", R);
    let _start_w_reduction = Instant::now();
    let mut w_dev = DeviceVec::<PolyRing>::device_malloc(R)?;
    let mut matcfg_w = MatMulConfig::default();
    matcfg_w.b_transposed = true;
    matrix_ops::matmul::<PolyRing>(
        &a_dev, 1, s_len as u32,
        &s_flat_ntt_dev, R as u32, s_len as u32,
        &matcfg_w,
        &mut w_dev,
    )?;
    negacyclic_ntt::ntt_inplace(&mut w_dev, NTTDir::kInverse, &ntt_cfg)?;
    // println!("  [DEBUG] Batched w-reduction finished.");
    // println!("  [TIMER] Prover w-reduction took: {:?}", start_w_reduction.elapsed());

    // Line 9: w_hat = G^{-1}_{b,r}(w)
    let w_hat_len = DELTA * R;
    let mut w_hat = DeviceVec::<PolyRing>::device_malloc(w_hat_len)?;
    balanced_decomposition::decompose::<PolyRing>(
        &w_dev[..], &mut w_hat[..], (1 << B1) as u32, &cfg,
    )?;

    // Line 10: v = D · w_hat
    let mut D_ntt = DeviceVec::<PolyRing>::device_malloc(pp.D.len())?;
    negacyclic_ntt::ntt(&pp.D, NTTDir::kForward, &ntt_cfg, &mut D_ntt[..])?;
    let mut w_hat_ntt = DeviceVec::<PolyRing>::device_malloc(w_hat.len())?;
    negacyclic_ntt::ntt(&w_hat, NTTDir::kForward, &ntt_cfg, &mut w_hat_ntt[..])?;
    let mut v_dev = DeviceVec::<PolyRing>::device_malloc(NROWS)?;
    matrix_ops::matmul::<PolyRing>(
        &D_ntt, NROWS as u32, D_COLS as u32,
        &w_hat_ntt, D_COLS as u32, 1,
        &MatMulConfig::default(), &mut v_dev,
    )?;
    negacyclic_ntt::ntt_inplace(&mut v_dev, NTTDir::kInverse, &ntt_cfg)?;
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

    // Lines 13 - 17
    // Build the Greyhound/Labrador witness z = [w_hat | t_hat | [s_1 | ... | s_r] c].
    let w_len = DELTA * R;
    let t_len = NROWS * DELTA * R;
    let mut z_full: Vec<PolyRing> = Vec::with_capacity(w_len + t_len + s_len);
    let _start_z_full_pack = Instant::now();

    let mut c_ntt = c;
    negacyclic_ntt::ntt_inplace(&mut c_ntt, NTTDir::kForward, &ntt_cfg)?;
    let mut folded_s_dev = DeviceVec::<PolyRing>::device_malloc(s_len)?;
    matrix_ops::matmul::<PolyRing>(
        &c_ntt, 1, R as u32,
        &s_flat_ntt_dev, R as u32, s_len as u32,
        &MatMulConfig::default(),
        &mut folded_s_dev,
    )?;
    negacyclic_ntt::ntt_inplace(&mut folded_s_dev, NTTDir::kInverse, &ntt_cfg)?;
    let mut folded_s = vec![PolyRing::zero(); s_len];
    folded_s_dev.copy_to_host(HostSlice::from_mut_slice(&mut folded_s))?;

    let mut t_hat_concat_host = vec![PolyRing::zero(); commit_return.t_hat_concat.len()];
    commit_return.t_hat_concat.copy_to_host(HostSlice::from_mut_slice(&mut t_hat_concat_host))?;

    z_full.extend_from_slice(&w_hat_host);
    z_full.extend_from_slice(&t_hat_concat_host);
    z_full.extend_from_slice(&folded_s);
    // println!("  [TIMER] Prover packing z_full blocks took: {:?}", start_z_full_pack.elapsed());
    
    // Build the constraint system: P matrix components + target h
    let constraint = labrador::ConstraintSystem {
        a_eval: a_host.clone(),
        b_eval: b_host.clone(),
        c_eval: c_host.clone(),
        sigma_inv_x: sigma_inv_x,
        A_ajtai: {
            let mut A_h = vec![PolyRing::zero(); pp.A.len()];
            pp.A.copy_to_host(HostSlice::from_mut_slice(&mut A_h))?;
            A_h
        },
        B_commit: {
            let mut B_h = vec![PolyRing::zero(); pp.B.len()];
            pp.B.copy_to_host(HostSlice::from_mut_slice(&mut B_h))?;
            B_h
        },
        D_commit: {
            let mut D_h = vec![PolyRing::zero(); pp.D.len()];
            pp.D.copy_to_host(HostSlice::from_mut_slice(&mut D_h))?;
            D_h
        },
        h_target: {
            let mut h = Vec::new();
            let mut u_host = vec![PolyRing::zero(); commit_return.u.len()];
            commit_return.u.copy_to_host(HostSlice::from_mut_slice(&mut u_host))?;
            h.extend_from_slice(&v_host);          // v (NROWS)
            h.extend_from_slice(&u_host);          // u (NROWS)
            h.push(sigma_inv_y_target);            // σ_{-1}(x)^-1 * y (1)
            h.push(PolyRing::zero());              // 0 (1)
            h.extend(std::iter::repeat(PolyRing::zero()).take(NROWS)); // 0 (NROWS)
            h
        },
        n_w: w_len,
        n_t: t_len,
        n_s: s_len,
        n_commit: NROWS,
    };

    let lab_witness = labrador::LabradorWitness {
        z: z_full,
        s: Vec::new(),
    };
    // println!("  [DEBUG] Labrador witness finished.");

    // Run Labrador prover (using shared CRS)
    let labrador_prove_start = Instant::now();
    let (labrador_proof, labrador_transcript) = labrador::labrador_prove(lab_crs, &lab_witness, &constraint)?;
    let labrador_prove_elapsed = labrador_prove_start.elapsed();
    // println!("  [DEBUG] Labrador prove finished.");
    Ok(ProverProof {
        y: y_ring,
        claimed_eval,
        v: v_host,
        narg_transcript,
        labrador_proof,
        labrador_transcript,
        labrador_prove_elapsed,
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
    lab_crs: &labrador::LabradorCRS,
) -> Result<bool, IcicleError> {
    Ok(eval_verifier_with_labrador_time(
        pp,
        commit_return,
        x_scalar,
        proof,
        lab_crs,
    )?.0)
}

pub fn eval_verifier_with_labrador_time(
    pp: &PublicParams,
    commit_return: &CommitReturn,
    x_scalar: &Zq,
    proof: &ProverProof,
    lab_crs: &labrador::LabradorCRS,
) -> Result<(bool, std::time::Duration), IcicleError> {
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

    let proof_eval = constant_term(&proof.y);
    if !zq_equal(&proof_eval, &proof.claimed_eval) {
        // println!("  [FAIL] Check 1: ct(y) != claimed f(x)");
        return Ok((false, std::time::Duration::ZERO));
    }
    // println!("  [PASS] Check 1: ct(y) matches claimed f(x)");

    let sigma_inv_y_target = sigma_inverse_apply(&proof.y, x_scalar);

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
        c_eval: c_host,
        sigma_inv_x,
        A_ajtai: {
            let mut A_h = vec![PolyRing::zero(); pp.A.len()];
            pp.A.copy_to_host(HostSlice::from_mut_slice(&mut A_h))?;
            A_h
        },
        B_commit: {
            let mut B_h = vec![PolyRing::zero(); pp.B.len()];
            pp.B.copy_to_host(HostSlice::from_mut_slice(&mut B_h))?;
            B_h
        },
        D_commit: {
            let mut D_h = vec![PolyRing::zero(); pp.D.len()];
            pp.D.copy_to_host(HostSlice::from_mut_slice(&mut D_h))?;
            D_h
        },
        h_target: {
            let mut u_host = vec![PolyRing::zero(); commit_return.u.len()];
            commit_return.u.copy_to_host(HostSlice::from_mut_slice(&mut u_host))?;
            let mut h = Vec::new();
            h.extend_from_slice(&proof.v);          // v (NROWS)
            h.extend_from_slice(&u_host);           // u (NROWS)
            h.push(sigma_inv_y_target);              // σ_{-1}(x)^-1 * y (1)
            h.push(PolyRing::zero());                // 0 (1)
            h.extend(std::iter::repeat(PolyRing::zero()).take(NROWS)); // 0 (NROWS)
            h
        },
        n_w: DELTA * R,
        n_t: NROWS * DELTA * R,
        n_s: DELTA0 * M,
        n_commit: NROWS,
    };

    // 4. Run Labrador Verifier — all SIS checks are inside
    let labrador_verify_start = Instant::now();
    let lab_ok = labrador::labrador_verify(
        lab_crs,
        &proof.labrador_proof,
        &proof.labrador_transcript,
        &constraint,
    )?;
    let labrador_verify_elapsed = labrador_verify_start.elapsed();
    if !lab_ok {
        // println!("  [FAIL] Check 2: Labrador verification failed");
        return Ok((false, labrador_verify_elapsed));
    }
    // println!("  [PASS] Check 2: Labrador verification passed");

    Ok((true, labrador_verify_elapsed))
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

fn constant_term(y: &PolyRing) -> Zq {
    let coeffs = unsafe {
        std::slice::from_raw_parts(y as *const _ as *const Zq, PolyRing::DEGREE)
    };
    coeffs[0]
}

fn zq_equal(a: &Zq, b: &Zq) -> bool {
    let size = std::mem::size_of::<Zq>();
    let a_bytes = unsafe { std::slice::from_raw_parts(a as *const _ as *const u8, size) };
    let b_bytes = unsafe { std::slice::from_raw_parts(b as *const _ as *const u8, size) };
    a_bytes == b_bytes
}

fn evaluate_polynomial(raw_f_coeffs: &[u32], x: &Zq) -> Zq {
    let mut y = Zq::zero();
    for coeff in raw_f_coeffs.iter().rev() {
        y = y * *x + Zq::from(*coeff);
    }
    y
}

fn seconds(duration: std::time::Duration) -> f64 {
    duration.as_secs_f64()
}

fn device_vec_to_host<T: Copy>(device_vec: &DeviceVec<T>) -> Vec<T> {
    let mut host_vec = vec![unsafe { std::mem::zeroed() }; device_vec.len()];
    device_vec.copy_to_host(HostSlice::from_mut_slice(&mut host_vec)).unwrap();
    host_vec
}

fn clone_device_vec<T: Copy>(device_vec: &DeviceVec<T>) -> DeviceVec<T> {
    DeviceVec::from_host_slice(&device_vec_to_host(device_vec))
}

fn clone_commit_return(commit_return: &CommitReturn) -> CommitReturn {
    CommitReturn {
        u: clone_device_vec(&commit_return.u),
        s: clone_device_vec(&commit_return.s),
        t_hat_concat: clone_device_vec(&commit_return.t_hat_concat),
    }
}

fn tamper_poly(poly: &mut PolyRing) {
    let d = PolyRing::DEGREE;
    let coeffs = unsafe { std::slice::from_raw_parts(poly as *const _ as *const Zq, d) };
    let mut tampered = coeffs.to_vec();
    tampered[0] = tampered[0] + Zq::one();
    *poly = PolyRing::from_slice(&tampered).unwrap();
}

fn poly_vec_bytes_equal(a: &[PolyRing], b: &[PolyRing]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let byte_len = a.len() * std::mem::size_of::<PolyRing>();
    let a_bytes = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const u8, byte_len) };
    let b_bytes = unsafe { std::slice::from_raw_parts(b.as_ptr() as *const u8, byte_len) };
    a_bytes == b_bytes
}

fn greyhound_challenge_host_with_params(
    params: GreyhoundTranscriptParams,
    u: &DeviceVec<PolyRing>,
    x: &Zq,
    y: &PolyRing,
    v: &DeviceVec<PolyRing>,
) -> Vec<PolyRing> {
    let domain_sep = DomainSeparator::new(FiatShamirGreyhound::protocol_id())
        .session(spongefish::session!("greyhound_pcs"))
        .instance(&[0u8; 0]);
    let mut prover_state = domain_sep.std_prover();
    let c = FiatShamirGreyhound::prove_with_params(&mut prover_state, params, u, x, y, v);
    device_vec_to_host(&c)
}

fn expect_reject(name: &str, accepted: bool) -> bool {
    if accepted {
        println!("  [TEST FAIL] {} accepted but should reject", name);
        false
    } else {
        println!("  [TEST PASS] {} rejected", name);
        true
    }
}

fn expect_change(name: &str, changed: bool) -> bool {
    if changed {
        println!("  [TEST PASS] {} changed", name);
        true
    } else {
        println!("  [TEST FAIL] {} did not change", name);
        false
    }
}

fn run_negative_tests(
    pp: &PublicParams,
    commit_return: &CommitReturn,
    x_scalar: &Zq,
    proof: &ProverProof,
    lab_crs: &labrador::LabradorCRS,
) -> Result<bool, IcicleError> {
    let mut failures = 0usize;

    println!("--------------------------------------------------");
    println!(" Running negative binding/tamper tests");

    let mut wrong_claim_proof = proof.clone();
    wrong_claim_proof.claimed_eval = wrong_claim_proof.claimed_eval + Zq::one();
    if !expect_reject(
        "Statement binding: claimed f(x)+1",
        eval_verifier(pp, commit_return, x_scalar, &wrong_claim_proof, lab_crs)?,
    ) {
        failures += 1;
    }

    {
        let v_dev = DeviceVec::from_host_slice(&proof.v);
        let base_c = greyhound_challenge_host_with_params(
            greyhound_transcript_params(),
            &commit_return.u,
            x_scalar,
            &proof.y,
            &v_dev,
        );
        let mut altered_params = greyhound_transcript_params();
        altered_params.m += 1;
        let altered_c = greyhound_challenge_host_with_params(
            altered_params,
            &commit_return.u,
            x_scalar,
            &proof.y,
            &v_dev,
        );
        if !expect_change(
            "Transcript binding: M parameter challenge",
            !poly_vec_bytes_equal(&base_c, &altered_c),
        ) {
            failures += 1;
        }
    }

    {
        let mut tampered_commit = clone_commit_return(commit_return);
        let mut u_host = device_vec_to_host(&tampered_commit.u);
        tamper_poly(&mut u_host[0]);
        tampered_commit.u = DeviceVec::from_host_slice(&u_host);
        if !expect_reject(
            "Tamper u[0]",
            eval_verifier(pp, &tampered_commit, x_scalar, proof, lab_crs)?,
        ) {
            failures += 1;
        }
    }

    {
        let mut tampered_proof = proof.clone();
        tamper_poly(&mut tampered_proof.v[0]);
        if !expect_reject(
            "Tamper v[0]",
            eval_verifier(pp, commit_return, x_scalar, &tampered_proof, lab_crs)?,
        ) {
            failures += 1;
        }
    }

    {
        let mut tampered_proof = proof.clone();
        tamper_poly(&mut tampered_proof.y);
        if !expect_reject(
            "Tamper proof.y",
            eval_verifier(pp, commit_return, x_scalar, &tampered_proof, lab_crs)?,
        ) {
            failures += 1;
        }
    }

    {
        let mut tampered_proof = proof.clone();
        if !tampered_proof.labrador_proof.z_folded.is_empty() {
            tamper_poly(&mut tampered_proof.labrador_proof.z_folded[0]);
        }
        if !expect_reject(
            "Tamper z_folded[0]",
            eval_verifier(pp, commit_return, x_scalar, &tampered_proof, lab_crs)?,
        ) {
            failures += 1;
        }
    }

    if failures == 0 {
        println!("  [TEST PASS] All negative tests passed");
        Ok(true)
    } else {
        println!("  [TEST FAIL] {} negative test(s) failed", failures);
        Ok(false)
    }
}

#[derive(Clone, Copy, Default)]
struct RunTimings {
    setup: f64,
    input: f64,
    lab_crs: f64,
    commit: f64,
    prover: f64,
    verifier: f64,
    total: f64,
    lab_prove: f64,
    lab_verify: f64,
    verified: bool,
}

impl RunTimings {
    fn greyhound_prover(&self) -> f64 {
        (self.prover - self.lab_prove).max(0.0)
    }

    fn greyhound_verifier(&self) -> f64 {
        (self.verifier - self.lab_verify).max(0.0)
    }

    fn greyhound_total(&self) -> f64 {
        self.setup + self.input + self.commit + self.greyhound_prover() + self.greyhound_verifier()
    }

    fn add_assign(&mut self, other: RunTimings) {
        self.setup += other.setup;
        self.input += other.input;
        self.lab_crs += other.lab_crs;
        self.commit += other.commit;
        self.prover += other.prover;
        self.verifier += other.verifier;
        self.total += other.total;
        self.lab_prove += other.lab_prove;
        self.lab_verify += other.lab_verify;
    }

    fn averaged(mut self, runs: usize) -> RunTimings {
        let denom = runs as f64;
        self.setup /= denom;
        self.input /= denom;
        self.lab_crs /= denom;
        self.commit /= denom;
        self.prover /= denom;
        self.verifier /= denom;
        self.total /= denom;
        self.lab_prove /= denom;
        self.lab_verify /= denom;
        self
    }
}

struct RunOutput {
    timings: RunTimings,
    pp: PublicParams,
    commit_ret: CommitReturn,
    x_eval: Zq,
    proof: ProverProof,
    lab_crs: labrador::LabradorCRS,
}

fn load_coeffs_csv(path: &str) -> Vec<u32> {
    let file = std::fs::File::open(path).expect("coefficient CSV open failed");
    let reader = std::io::BufReader::new(file);
    let mut coeffs = Vec::with_capacity(N);

    for line in std::io::BufRead::lines(reader) {
        let line = line.expect("coefficient CSV read failed");
        for token in line.split(',') {
            let trimmed = token.trim();
            if trimmed.is_empty() {
                continue;
            }
            coeffs.push(trimmed.parse::<u32>().expect("invalid coefficient in CSV"));
        }
    }

    assert_eq!(coeffs.len(), N, "coefficient CSV must contain exactly N entries");
    coeffs
}

fn greyhound_norm_bound_sq() -> f64 {
    let d_f64 = PolyRing::DEGREE as f64;
    let b0_f64 = (1u64 << B0) as f64;
    let b1_f64 = (1u64 << B1) as f64;
    // TODO: Magic number. This constant is problematic. In the labrador paper as far as I can tell, 
    // this number is the norm bound of the challenge vector c , which features very strict coefficients
    // as stated in section 2, challenge space. I need to look into this more and use trict coeffs
    // instead of random sampling icicle func
    let kappa = 51.0;

    let term1 = b1_f64.powi(2) * (NROWS as f64 + 1.0) * (DELTA as f64) * (R as f64) * d_f64;
    let term2 = ((R as f64) * kappa * b0_f64).powi(2) * (DELTA0 as f64) * (M as f64) * d_f64;
    term1 + term2
}

fn print_timing_summary(title: &str, timings: &RunTimings) {
    println!("--------------------------------------------------");
    println!("{}", title);
    println!("  setup():              {} s", timings.setup);
    println!("  input load/generate:  {} s", timings.input);
    println!("  Labrador CRS setup:   {} s", timings.lab_crs);
    println!("  commit():             {} s", timings.commit);
    println!("  eval_prover():        {} s", timings.prover);
    println!("  eval_verifier():      {} s", timings.verifier);
    println!("  total:                {} s", timings.total);
}

fn print_greyhound_runtime_summary(title: &str, timings: &RunTimings) {
    println!("--------------------------------------------------");
    println!("{}", title);
    println!("  setup():              {} s", timings.setup);
    println!("  input load/generate:  {} s", timings.input);
    println!("  commit():             {} s", timings.commit);
    println!("  eval_prover():        {} s", timings.greyhound_prover());
    println!("  eval_verifier():      {} s", timings.greyhound_verifier());
    println!("  total:                {} s", timings.greyhound_total());
    println!("  excluded Labrador CRS setup:  {} s", timings.lab_crs);
    println!("  excluded Labrador prove:      {} s", timings.lab_prove);
    println!("  excluded Labrador verify:     {} s", timings.lab_verify);
}

fn run_pipeline_once(
    coeff_path: Option<&str>,
    seed: u64,
    print_verdict: bool,
) -> Result<RunOutput, IcicleError> {
    let total_start = Instant::now();

    let setup_start = Instant::now();
    let pp = setup();
    let setup_elapsed = setup_start.elapsed();

    let gamma_bar_sq = greyhound_norm_bound_sq();

    let lab_crs_start = Instant::now();
    let lab_crs = labrador::LabradorCRS::setup(
        R, D_COLS + B_COLS + A_COLS,
        NROWS, N1, N1,
        DELTA, DELTA,
        (1 << B1) as u32, (1 << B1) as u32, (1 << B0) as u32, // b, b1, b2
        1, 1, gamma_bar_sq,
    );
    let lab_crs_elapsed = lab_crs_start.elapsed();

    let input_start = Instant::now();
    let raw_f_coeffs: Vec<u32> = if let Some(path) = coeff_path {
        load_coeffs_csv(path)
    } else {
        let mut coeff_rng = ChaCha20Rng::seed_from_u64(seed);
        (0..N).map(|_| coeff_rng.r#gen::<u32>()).collect()
    };
    let input_elapsed = input_start.elapsed();

    let commit_start = Instant::now();
    let commit_ret = commit(&raw_f_coeffs, &pp).expect("Commit failed");
    let commit_elapsed = commit_start.elapsed();

    let mut rng = ChaCha20Rng::seed_from_u64(seed ^ 0xa076_1d64_78bd_642f);
    let x_eval = Zq::from(rng.r#gen::<u32>());

    let prover_start = Instant::now();
    let proof = eval_prover(&pp, &raw_f_coeffs, &commit_ret, &x_eval, &lab_crs)?;
    let prover_elapsed = prover_start.elapsed();
    
    let verifier_start = Instant::now();
    let (verified, labrador_verify_elapsed) = eval_verifier_with_labrador_time(
        &pp,
        &commit_ret,
        &x_eval,
        &proof,
        &lab_crs,
    )?;
    let verifier_elapsed = verifier_start.elapsed();

    if print_verdict {
        if verified {
            println!("Verifier PASSED successfully!");
        } else {
            println!("Verifier REJECTED the proof!");
        }
    }

    let total_elapsed = total_start.elapsed();
    let timings = RunTimings {
        setup: seconds(setup_elapsed),
        input: seconds(input_elapsed),
        lab_crs: seconds(lab_crs_elapsed),
        commit: seconds(commit_elapsed),
        prover: seconds(prover_elapsed),
        verifier: seconds(verifier_elapsed),
        total: seconds(total_elapsed),
        lab_prove: seconds(proof.labrador_prove_elapsed),
        lab_verify: seconds(labrador_verify_elapsed),
        verified,
    };

    Ok(RunOutput {
        timings,
        pp,
        commit_ret,
        x_eval,
        proof,
        lab_crs,
    })
}

fn main() {
    // println!("Loading default backend and initializing device...");
    let _ = icicle_runtime::runtime::load_backend("/workspace/icicle").unwrap();
    //let device = icicle_runtime::Device::new("CPU", 0);
    let device = icicle_runtime::Device::new("CUDA", 0);
    icicle_runtime::set_device(&device).unwrap();

    let args: Vec<String> = std::env::args().collect();
    let mut run_tests = false;
    let mut runtime_only = false;
    let mut comparison = false;
    let mut coeff_path: Option<String> = None;
    for arg in args.iter().skip(1) {
        if arg == "--test" {
            run_tests = true;
        } else if arg == "--runtime" {
            runtime_only = true;
        } else if arg == "--comparison" {
            comparison = true;
        } else if coeff_path.is_none() {
            coeff_path = Some(arg.clone());
        } else {
            panic!("usage: greyhound [--test] [--runtime] [--comparison] [coeffs.csv]");
        }
    }

    if comparison {
        const COMPARISON_RUNS: usize = 10;
        let mut totals = RunTimings::default();
        let mut passed = 0usize;
        for run in 0..COMPARISON_RUNS {
            println!("Comparison run {}/{}...", run + 1, COMPARISON_RUNS);
            let output = run_pipeline_once(
                coeff_path.as_deref(),
                12345 + run as u64,
                false,
            ).expect("comparison run failed");
            if output.timings.verified {
                passed += 1;
            }
            totals.add_assign(output.timings);
        }

        if passed == COMPARISON_RUNS {
            println!("Verifier PASSED successfully! ({}/{})", passed, COMPARISON_RUNS);
        } else {
            println!("Verifier REJECTED at least one proof! ({}/{})", passed, COMPARISON_RUNS);
        }

        let averages = totals.averaged(COMPARISON_RUNS);
        print_timing_summary("Average timing summary (--comparison, 10 runs):", &averages);
        print_greyhound_runtime_summary("Average Greyhound-only runtime summary (--comparison, 10 runs):", &averages);

        if passed != COMPARISON_RUNS {
            std::process::exit(1);
        }
        return;
    }

    let output = run_pipeline_once(coeff_path.as_deref(), 12345, true).expect("run failed");

    let mut tests_ok = true;
    if run_tests {
        tests_ok = run_negative_tests(
            &output.pp,
            &output.commit_ret,
            &output.x_eval,
            &output.proof,
            &output.lab_crs,
        ).expect("negative tests failed to run");
    }

    print_timing_summary("Timing summary:", &output.timings);

    if runtime_only {
        print_greyhound_runtime_summary("Greyhound-only runtime summary (--runtime):", &output.timings);
    }

    if !output.timings.verified || !tests_ok {
        std::process::exit(1);
    }
}


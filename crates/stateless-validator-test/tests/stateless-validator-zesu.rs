//! Execution tests for `stateless-validator-zesu` guest program.
//!
//! These tests require pre-built Zesu ELFs downloaded from zesu-zkvm releases.
//! Set the appropriate env var before running; tests are skipped silently when unset.
//!
//!   ZESU_ELF_ZISK   — path to `stateless-validator-zesu-zisk.elf`
//!                     Raw SSZ bytes are passed directly via `Input::with_stdin`.
//!
//! Expected output format: `[new_payload_request_root (32B)][success (1B)][chain_id (8B)]`
//! padded to 256 bytes for ZisK.

use std::path::PathBuf;

use alloy_primitives::{Bytes, hex};
use ere_dockerized::zkVMKind;
use libssz::SszEncode;
use stateless::ExecutionWitness;
use stateless_validator_reth::guest::{
    Guest, StatelessValidatorRethGuest, StatelessValidatorRethInput,
    new_payload_request::NewPayloadRequest,
};
use stateless_validator_test::{
    NoopPlatform, TestCase, get_fixtures, stateless_validator::get_stateless_validator_output,
};

// ── SSZ encoding helpers ───────────────────────────────────────────────────────

/// Encode a `List[ByteList]` as SSZ: N×4-byte LE offset table then element bytes.
fn encode_byte_list_list(items: &[impl AsRef<[u8]>]) -> Vec<u8> {
    if items.is_empty() {
        return vec![];
    }
    let n = items.len();
    let mut buf = Vec::new();
    let mut offset = (n * 4) as u32;
    for item in items {
        buf.extend_from_slice(&offset.to_le_bytes());
        offset += item.as_ref().len() as u32;
    }
    for item in items {
        buf.extend_from_slice(item.as_ref());
    }
    buf
}

/// Encode an `ExecutionWitness` to the `SszExecutionWitness` format:
/// 12-byte fixed region `[off_state, off_codes, off_headers]` + variable data.
fn encode_witness(witness: &ExecutionWitness) -> Vec<u8> {
    let state_bytes =
        encode_byte_list_list(&witness.state.iter().map(Bytes::as_ref).collect::<Vec<_>>());
    let codes_bytes =
        encode_byte_list_list(&witness.codes.iter().map(Bytes::as_ref).collect::<Vec<_>>());
    let headers_bytes =
        encode_byte_list_list(&witness.headers.iter().map(Bytes::as_ref).collect::<Vec<_>>());

    let off_state: u32 = 12;
    let off_codes: u32 = off_state + state_bytes.len() as u32;
    let off_headers: u32 = off_codes + codes_bytes.len() as u32;

    let mut buf = Vec::with_capacity(
        12 + state_bytes.len() + codes_bytes.len() + headers_bytes.len(),
    );
    buf.extend_from_slice(&off_state.to_le_bytes());
    buf.extend_from_slice(&off_codes.to_le_bytes());
    buf.extend_from_slice(&off_headers.to_le_bytes());
    buf.extend_from_slice(&state_bytes);
    buf.extend_from_slice(&codes_bytes);
    buf.extend_from_slice(&headers_bytes);
    buf
}

/// Mainnet Prague activation timestamp. Zesu's `mainnetSpec(block_number, timestamp)`
/// returns a blob-capable spec (≥ .prague) for any timestamp at or above this value.
/// Below it, `mainnetSpec` returns a pre-Cancun spec, which rejects blob transactions.
const MAINNET_PRAGUE_TIME: u64 = 1_746_612_311;

/// Returns the timestamp from a supported `NewPayloadRequest` variant.
fn payload_timestamp(input: &StatelessValidatorRethInput) -> u64 {
    match &input.new_payload_request {
        NewPayloadRequest::Amsterdam(inner) => inner.execution_payload.timestamp,
        NewPayloadRequest::ElectraFulu(inner) => inner.execution_payload.timestamp,
        _ => panic!("only Prague/Osaka (V3 EP) and Amsterdam (V4 EP) are supported"),
    }
}

/// Returns true when the extended 24-byte SSZ layout with `fork_name` is needed.
///
/// Zesu's `specForBlock` only recognises "Prague", "Osaka", and "Amsterdam" — there is
/// no "Bpo2" string. For real mainnet blocks (chain_id 0/1, timestamp ≥ MAINNET_PRAGUE_TIME)
/// `mainnetSpec` returns the correct spec including BPO1/BPO2 upgrades, so we use the
/// standard 20-byte layout and let Zesu call `mainnetSpec` itself.
///
/// We use the extended layout only when `mainnetSpec` would fail: synthetic fixtures with
/// an artificial timestamp below prague activation, or non-mainnet chains.
fn needs_fork_name(input: &StatelessValidatorRethInput) -> bool {
    let chain_id = input.chain_config.chain_id;
    let timestamp = payload_timestamp(input);
    let is_mainnet = chain_id == 0 || chain_id == 1;
    !is_mainnet || timestamp < MAINNET_PRAGUE_TIME
}

/// Fork name for the extended layout. Only called when `needs_fork_name` returns true.
///
/// Zesu's `specForBlock` maps "Prague" → .prague, "Osaka" → .osaka, "Amsterdam" → .amsterdam.
fn zesu_fork_name(input: &StatelessValidatorRethInput) -> &'static str {
    let timestamp = payload_timestamp(input);
    let cc = &input.chain_config;
    if cc.amsterdam_time.is_some_and(|t| timestamp >= t) {
        return "Amsterdam";
    }
    if cc.osaka_time.is_some_and(|t| timestamp >= t) {
        return "Osaka";
    }
    "Prague"
}

/// Encode the NPR bytes from a supported `NewPayloadRequest` variant.
fn encode_npr(input: &StatelessValidatorRethInput) -> Vec<u8> {
    match &input.new_payload_request {
        NewPayloadRequest::Amsterdam(inner) => inner.to_ssz(),
        NewPayloadRequest::ElectraFulu(inner) => inner.to_ssz(),
        _ => panic!("only Prague/Osaka (V3 EP) and Amsterdam (V4 EP) are supported"),
    }
}

/// Encode `StatelessValidatorRethInput` to Zesu's `SszStatelessInput` format.
///
/// Uses the standard 20-byte fixed region for real mainnet blocks (letting Zesu call
/// `mainnetSpec` for correct BPO1/BPO2 detection), and the extended 24-byte fixed region
/// with a `fork_name` field for synthetic or non-mainnet fixtures where `mainnetSpec`
/// would return a pre-blob spec.
///
/// `public_keys` is always encoded as empty — Zesu performs ecrecover internally.
fn encode_zesu_ssz(input: &StatelessValidatorRethInput) -> Vec<u8> {
    let npr_bytes = encode_npr(input);
    let witness_bytes = encode_witness(&input.witness);
    let empty: &[&[u8]] = &[];
    let pubkeys_bytes = encode_byte_list_list(empty);
    let chain_id: u64 = input.chain_config.chain_id;

    if needs_fork_name(input) {
        let fork_name_bytes = zesu_fork_name(input).as_bytes();

        // Extended fixed region (24 bytes):
        //   [0..4]   offset → new_payload_request (= 24, signals extended layout to Zesu)
        //   [4..8]   offset → witness
        //   [8..16]  chain_id (u64 LE, inline)
        //   [16..20] offset → public_keys
        //   [20..24] offset → fork_name (ByteList, variable)
        let off_npr: u32 = 24;
        let off_witness: u32 = off_npr + npr_bytes.len() as u32;
        let off_pubkeys: u32 = off_witness + witness_bytes.len() as u32;
        let off_fork_name: u32 = off_pubkeys + pubkeys_bytes.len() as u32;

        let mut buf = Vec::with_capacity(
            24 + npr_bytes.len()
                + witness_bytes.len()
                + pubkeys_bytes.len()
                + fork_name_bytes.len(),
        );
        buf.extend_from_slice(&off_npr.to_le_bytes());
        buf.extend_from_slice(&off_witness.to_le_bytes());
        buf.extend_from_slice(&chain_id.to_le_bytes());
        buf.extend_from_slice(&off_pubkeys.to_le_bytes());
        buf.extend_from_slice(&off_fork_name.to_le_bytes());
        buf.extend_from_slice(&npr_bytes);
        buf.extend_from_slice(&witness_bytes);
        buf.extend_from_slice(&pubkeys_bytes);
        buf.extend_from_slice(fork_name_bytes);
        buf
    } else {
        // Standard fixed region (20 bytes): Zesu calls mainnetSpec for spec detection.
        //   [0..4]   offset → new_payload_request (= 20)
        //   [4..8]   offset → witness
        //   [8..16]  chain_id (u64 LE, inline)
        //   [16..20] offset → public_keys
        let off_npr: u32 = 20;
        let off_witness: u32 = off_npr + npr_bytes.len() as u32;
        let off_pubkeys: u32 = off_witness + witness_bytes.len() as u32;

        let mut buf =
            Vec::with_capacity(20 + npr_bytes.len() + witness_bytes.len() + pubkeys_bytes.len());
        buf.extend_from_slice(&off_npr.to_le_bytes());
        buf.extend_from_slice(&off_witness.to_le_bytes());
        buf.extend_from_slice(&chain_id.to_le_bytes());
        buf.extend_from_slice(&off_pubkeys.to_le_bytes());
        buf.extend_from_slice(&npr_bytes);
        buf.extend_from_slice(&witness_bytes);
        buf.extend_from_slice(&pubkeys_bytes);
        buf
    }
}

// ── Test driver ───────────────────────────────────────────────────────────────

/// Run all fixtures (valid and invalid) against a pre-built Zesu ELF in a single container
/// instance.
///
/// Valid fixtures: expected root comes from the independent precomputed map.
/// Invalid fixtures: the guest should set success=false; expected root is derived host-side
/// via `StatelessValidatorRethGuest` (independent of Zesu's hash computation).
fn test_execution(zkvm_kind: zkVMKind, elf_env_var: &str) {
    let Some(elf_path) = std::env::var(elf_env_var).ok().map(PathBuf::from) else {
        eprintln!("Skipping: {elf_env_var} is not set");
        return;
    };

    let fixtures = get_fixtures();
    let test_cases = fixtures.into_iter().map(|fixture| {
        let reth_input =
            StatelessValidatorRethInput::new(&fixture.stateless_input, fixture.success).unwrap();

        let encoded = encode_zesu_ssz(&reth_input);

        let output = if fixture.success {
            get_stateless_validator_output(
                fixture.stateless_input.block.hash_slow(),
                fixture.success,
                fixture.stateless_input.chain_config.chain_id,
            )
        } else {
            // For invalid blocks the precomputed map has no entry; derive host-side instead.
            StatelessValidatorRethGuest::compute::<NoopPlatform>(reth_input.clone())
        };

        // Zesu output: [root (32B)][success (1B)][chain_id (8B)]
        // Match ssz.zig: chain_id = 0 is treated as mainnet (1).
        let chain_id = {
            let id = fixture.stateless_input.chain_config.chain_id;
            if id == 0 { 1 } else { id }
        };
        let mut expected = Vec::with_capacity(41);
        expected.extend_from_slice(&output.new_payload_request_root);
        expected.push(fixture.success as u8);
        expected.extend_from_slice(&chain_id.to_le_bytes());

        TestCase::from_raw(fixture.name, encoded, expected)
    });

    stateless_validator_test::test_execution_from_elf(&elf_path, zkvm_kind, test_cases);
}

#[test]
fn test_execution_zisk() {
    test_execution(zkVMKind::Zisk, "ZESU_ELF_ZISK");
}

// ── Diagnostic (no Docker required) ──────────────────────────────────────────

/// Verifies the host-side SSZ encoding and expected-output pipeline without running ZisK.
///
/// For each fixture this checks:
/// - `encode_zesu_ssz` produces well-formed SSZ (valid fixed-region offsets, embedded chain_id)
/// - The expected output bytes are correctly assembled for both valid and invalid blocks
/// - No panics in the reth host-side path for invalid blocks
///
/// Run with: `cargo test --package stateless-validator-test --test stateless-validator-zesu
///            -- encode_pipeline --nocapture`
#[test]
fn test_encode_pipeline() {
    let fixtures = get_fixtures();
    assert!(!fixtures.is_empty(), "no fixtures loaded");

    let mut ok = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for fixture in &fixtures {
        let reth_input =
            match StatelessValidatorRethInput::new(&fixture.stateless_input, fixture.success) {
                Ok(i) => i,
                Err(e) => {
                    failures.push(format!("[{}] StatelessValidatorRethInput::new failed: {e}", fixture.name));
                    continue;
                }
            };

        let encoded = encode_zesu_ssz(&reth_input);

        // ── SSZ structure ──────────────────────────────────────────────────
        if encoded.len() < 20 {
            failures.push(format!("[{}] encoded SSZ too short: {} bytes", fixture.name, encoded.len()));
            continue;
        }
        let off_npr = u32::from_le_bytes(encoded[0..4].try_into().unwrap());
        let expected_off = if needs_fork_name(&reth_input) { 24u32 } else { 20u32 };
        if off_npr != expected_off {
            failures.push(format!(
                "[{}] bad NPR offset: got {off_npr}, expected {expected_off}",
                fixture.name
            ));
        }
        let embedded_chain_id = u64::from_le_bytes(encoded[8..16].try_into().unwrap());
        if embedded_chain_id != fixture.stateless_input.chain_config.chain_id {
            failures.push(format!(
                "[{}] chain_id mismatch in SSZ: got {embedded_chain_id}, expected {}",
                fixture.name, fixture.stateless_input.chain_config.chain_id
            ));
        }

        // ── Expected output ────────────────────────────────────────────────
        let output = if fixture.success {
            get_stateless_validator_output(
                fixture.stateless_input.block.hash_slow(),
                fixture.success,
                fixture.stateless_input.chain_config.chain_id,
            )
        } else {
            StatelessValidatorRethGuest::compute::<NoopPlatform>(reth_input.clone())
        };

        let effective_chain_id = if fixture.stateless_input.chain_config.chain_id == 0 {
            1u64
        } else {
            fixture.stateless_input.chain_config.chain_id
        };
        let mut expected_bytes = Vec::with_capacity(41);
        expected_bytes.extend_from_slice(&output.new_payload_request_root);
        expected_bytes.push(fixture.success as u8);
        expected_bytes.extend_from_slice(&effective_chain_id.to_le_bytes());

        if expected_bytes.len() != 41 {
            failures.push(format!("[{}] expected output wrong length: {}", fixture.name, expected_bytes.len()));
        }
        if output.successful_block_validation != fixture.success {
            failures.push(format!(
                "[{}] host-side success mismatch: reth says {}, fixture says {}",
                fixture.name, output.successful_block_validation, fixture.success
            ));
        }

        eprintln!(
            "[{}] success={} chain_id={} ssz={}B off_npr={} root={} fork_name={}",
            fixture.name,
            fixture.success,
            effective_chain_id,
            encoded.len(),
            off_npr,
            hex::encode(&output.new_payload_request_root[..8]),
            if needs_fork_name(&reth_input) { zesu_fork_name(&reth_input) } else { "mainnetSpec" },
        );

        ok += 1;
    }

    eprintln!("\n{ok}/{} fixtures encoded OK", fixtures.len());
    if !failures.is_empty() {
        for f in &failures {
            eprintln!("FAIL: {f}");
        }
        panic!("{} fixture(s) failed encoding validation", failures.len());
    }
}

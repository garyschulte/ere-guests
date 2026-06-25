//! Execution tests for `stateless-validator-zesu` guest program.
//!
//! These tests require pre-built Zesu ELFs downloaded from zesu-zkvm releases.
//! Set the appropriate env var before running; tests are skipped silently when unset.
//!
//!   ZESU_ELF_ZISK   — path to `stateless-validator-zesu-zisk.elf`
//!                     Raw SSZ bytes are passed directly via `Input::with_stdin`.
//!
//! Input format (v0.4.1 SszStatelessInput):
//!   `[0x00, 0x01][off_npr(4)][off_witness(4)][off_chain_config(4)][off_pubkeys(4)]`
//!   followed by variable fields: [SszNewPayloadRequest][SszExecutionWitness][SszChainConfig][pubkeys]
//!
//! Expected output format (v0.4.1, 105 bytes):
//!   `[new_payload_request_root (32B)][success (1B)][SszChainConfig (72B)]`
//! padded to 256 bytes for ZisK.
//!
//! Note: expected roots are computed by reth's `hash_tree_root` on the host side.  Both
//! zesu and reth implement the same spec-compliant SSZ schema (ByteList[2^30] for
//! `block_access_list`, depth 25), so their `new_payload_request_root` values agree.

use std::path::PathBuf;

use alloy_primitives::{Bytes, hex};
use ere_dockerized::zkVMKind;
use libssz::SszEncode;
use stateless::ExecutionWitness;
use stateless_validator_common::new_payload_request::NativeSha256Hasher;
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

/// Returns the timestamp from a supported `NewPayloadRequest` variant.
fn payload_timestamp(input: &StatelessValidatorRethInput) -> u64 {
    match &input.new_payload_request {
        NewPayloadRequest::Amsterdam(inner) => inner.execution_payload.timestamp,
        NewPayloadRequest::ElectraFulu(inner) => inner.execution_payload.timestamp,
        _ => panic!("only Prague/Osaka (V3 EP) and Amsterdam (V4 EP) are supported"),
    }
}

/// Returns the zesu ProtocolFork enum index for the active fork.
///
/// Zesu's decoder reads this index and maps it via `forkNameFromIndex`:
///   17 → "Prague", 18 → "Osaka", 24 → "Amsterdam"
fn zesu_fork_idx(input: &StatelessValidatorRethInput) -> u64 {
    let timestamp = payload_timestamp(input);
    let cc = &input.chain_config;
    if cc.amsterdam_time.is_some_and(|t| timestamp >= t) {
        return 24;
    }
    if cc.osaka_time.is_some_and(|t| timestamp >= t) {
        return 18;
    }
    17
}

/// Encode the v0.4.1 `SszChainConfig` input container (20 bytes):
///   [0..8]   chain_id (uint64 LE)
///   [8..12]  off_active_fork = 12 (uint32 LE, points to fork_idx below)
///   [12..20] fork_idx (uint64 LE, ProtocolFork enum index)
fn encode_chain_config(chain_id: u64, fork_idx: u64) -> Vec<u8> {
    let mut buf = Vec::with_capacity(20);
    buf.extend_from_slice(&chain_id.to_le_bytes());
    buf.extend_from_slice(&12u32.to_le_bytes());
    buf.extend_from_slice(&fork_idx.to_le_bytes());
    buf
}

/// Encode the NPR bytes from a supported `NewPayloadRequest` variant.
fn encode_npr(input: &StatelessValidatorRethInput) -> Vec<u8> {
    match &input.new_payload_request {
        NewPayloadRequest::Amsterdam(inner) => inner.to_ssz(),
        NewPayloadRequest::ElectraFulu(inner) => inner.to_ssz(),
        _ => panic!("only Prague/Osaka (V3 EP) and Amsterdam (V4 EP) are supported"),
    }
}

/// Encode `StatelessValidatorRethInput` to Zesu's v0.4.1 `SszStatelessInput` format.
///
/// Layout:
///   [0..2]   schema_id = 0x0001 (BE)
///   [2..6]   off_npr = 16  (must equal the 4-offset fixed-region size; decoder asserts this)
///   [6..10]  off_witness
///   [10..14] off_chain_config
///   [14..18] off_pubkeys
///   [18..]   variable: [SszNewPayloadRequest][SszExecutionWitness][SszChainConfig][pubkeys]
///
/// `public_keys` is always empty — Zesu performs ecrecover internally.
fn encode_zesu_ssz(input: &StatelessValidatorRethInput) -> Vec<u8> {
    let npr_bytes = encode_npr(input);
    let witness_bytes = encode_witness(&input.witness);
    let empty: &[&[u8]] = &[];
    let pubkeys_bytes = encode_byte_list_list(empty);
    let chain_id = input.chain_config.chain_id;
    let fork_idx = zesu_fork_idx(input);
    let chain_config_bytes = encode_chain_config(chain_id, fork_idx);

    // Offsets are relative to the body (after the 2-byte schema_id).
    let off_npr: u32 = 16;
    let off_witness: u32 = off_npr + npr_bytes.len() as u32;
    let off_chain_config: u32 = off_witness + witness_bytes.len() as u32;
    let off_pubkeys: u32 = off_chain_config + chain_config_bytes.len() as u32;

    let total = 2 + 16
        + npr_bytes.len()
        + witness_bytes.len()
        + chain_config_bytes.len()
        + pubkeys_bytes.len();
    let mut buf = Vec::with_capacity(total);
    buf.push(0x00);
    buf.push(0x01);
    buf.extend_from_slice(&off_npr.to_le_bytes());
    buf.extend_from_slice(&off_witness.to_le_bytes());
    buf.extend_from_slice(&off_chain_config.to_le_bytes());
    buf.extend_from_slice(&off_pubkeys.to_le_bytes());
    buf.extend_from_slice(&npr_bytes);
    buf.extend_from_slice(&witness_bytes);
    buf.extend_from_slice(&chain_config_bytes);
    buf.extend_from_slice(&pubkeys_bytes);
    buf
}

// ── Expected output constant ───────────────────────────────────────────────────

/// Zesu v0.4.1 hardcodes this 72-byte `SszChainConfig` into bytes [33..105] of every
/// output, following `new_payload_request_root (32B)` and `success (1B)`.
///
/// Encodes: chain_id=1 (mainnet), active_fork=Amsterdam (index 24), activation
/// timestamp=0, EIP-7691 blob schedule (target=14, max=21, fraction=0xB24B3F).
const SSZ_CHAIN_CONFIG_AMSTERDAM_MAINNET: [u8; 72] = [
    0x25, 0x00, 0x00, 0x00, // offset to chain_config body = 37
    0x01, 0x00, 0x00, 0x00, // chain_id = 1 (low 4 bytes)
    0x00, 0x00, 0x00, 0x00, // chain_id (high 4 bytes)
    0x0c, 0x00, 0x00, 0x00, // offset to active_fork = 12
    0x18, 0x00, 0x00, 0x00, // fork = 24 (Amsterdam, low 4 bytes)
    0x00, 0x00, 0x00, 0x00, // fork (high 4 bytes)
    0x10, 0x00, 0x00, 0x00, // offset to activation = 16
    0x20, 0x00, 0x00, 0x00, // offset to blob_schedule = 32
    0x08, 0x00, 0x00, 0x00, // activation.block_number offset = 8 (empty)
    0x08, 0x00, 0x00, 0x00, // activation.timestamp offset = 8
    0x00, 0x00, 0x00, 0x00, // activation.timestamp[0] = 0 (low 4 bytes)
    0x00, 0x00, 0x00, 0x00, // activation.timestamp[0] (high 4 bytes)
    0x0e, 0x00, 0x00, 0x00, // blob_schedule[0].target = 14 (low 4 bytes)
    0x00, 0x00, 0x00, 0x00, // blob_schedule[0].target (high 4 bytes)
    0x15, 0x00, 0x00, 0x00, // blob_schedule[0].max = 21 (low 4 bytes)
    0x00, 0x00, 0x00, 0x00, // blob_schedule[0].max (high 4 bytes)
    0x3f, 0x4b, 0xb2, 0x00, // blob_schedule[0].base_fee_update_fraction = 0xB24B3F
    0x00, 0x00, 0x00, 0x00, // (high 4 bytes)
];

// ── Test driver ───────────────────────────────────────────────────────────────

/// Run all fixtures (valid and invalid) against a pre-built Zesu ELF in a single container
/// instance.
///
/// Expected roots are computed by reth's `hash_tree_root` on the host side.  With zesu's
/// `block_access_list` Merkle depth corrected to ByteList[2^30] (depth 25, matching the
/// spec), zesu and reth produce identical `new_payload_request_root` values.
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

        let root = reth_input.new_payload_request.tree_hash_root(&NativeSha256Hasher);

        // Zesu v0.4.1 output: [root(32)][success(1)][SszChainConfig(72)] = 105 bytes.
        // ZisK pads to 256 bytes in the framework.
        let mut expected = Vec::with_capacity(105);
        expected.extend_from_slice(&root);
        expected.push(fixture.success as u8);
        expected.extend_from_slice(&SSZ_CHAIN_CONFIG_AMSTERDAM_MAINNET);

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
/// - `encode_zesu_ssz` produces v0.4.1 SSZ (schema_id 0x0001, off_npr=16, chain_id in chain_config)
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

        // ── v0.4.1 SSZ structure ───────────────────────────────────────────
        if encoded.len() < 18 {
            failures.push(format!("[{}] encoded SSZ too short: {} bytes", fixture.name, encoded.len()));
            continue;
        }
        if encoded[0] != 0x00 || encoded[1] != 0x01 {
            failures.push(format!(
                "[{}] bad schema_id: got [{:#04x}, {:#04x}], expected [0x00, 0x01]",
                fixture.name, encoded[0], encoded[1]
            ));
        }
        let off_npr = u32::from_le_bytes(encoded[2..6].try_into().unwrap());
        if off_npr != 16 {
            failures.push(format!(
                "[{}] bad NPR offset: got {off_npr}, expected 16",
                fixture.name
            ));
        }
        let off_witness = u32::from_le_bytes(encoded[6..10].try_into().unwrap());
        let off_chain_config = u32::from_le_bytes(encoded[10..14].try_into().unwrap());
        let off_pubkeys = u32::from_le_bytes(encoded[14..18].try_into().unwrap());
        // chain_id is the first 8 bytes of chain_config; body starts at encoded[2]
        let cc_start = 2 + off_chain_config as usize;
        if cc_start + 8 > encoded.len() {
            failures.push(format!("[{}] chain_config extends past end of encoded buffer", fixture.name));
        } else {
            let embedded_chain_id =
                u64::from_le_bytes(encoded[cc_start..cc_start + 8].try_into().unwrap());
            if embedded_chain_id != fixture.stateless_input.chain_config.chain_id {
                failures.push(format!(
                    "[{}] chain_id mismatch in SszChainConfig: got {embedded_chain_id}, expected {}",
                    fixture.name, fixture.stateless_input.chain_config.chain_id
                ));
            }
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

        let mut expected_bytes = Vec::with_capacity(105);
        expected_bytes.extend_from_slice(&output.new_payload_request_root);
        expected_bytes.push(fixture.success as u8);
        expected_bytes.extend_from_slice(&SSZ_CHAIN_CONFIG_AMSTERDAM_MAINNET);

        if expected_bytes.len() != 105 {
            failures.push(format!("[{}] expected output wrong length: {}", fixture.name, expected_bytes.len()));
        }
        if output.successful_block_validation != fixture.success {
            failures.push(format!(
                "[{}] host-side success mismatch: reth says {}, fixture says {}",
                fixture.name, output.successful_block_validation, fixture.success
            ));
        }

        let fork_idx = zesu_fork_idx(&reth_input);
        eprintln!(
            "[{}] success={} chain_id={} ssz={}B off_npr={} off_witness={} off_chain_config={} off_pubkeys={} fork_idx={} root={}",
            fixture.name,
            fixture.success,
            fixture.stateless_input.chain_config.chain_id,
            encoded.len(),
            off_npr,
            off_witness,
            off_chain_config,
            off_pubkeys,
            fork_idx,
            hex::encode(&output.new_payload_request_root[..8]),
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

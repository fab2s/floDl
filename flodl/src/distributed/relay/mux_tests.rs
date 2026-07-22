//! Round-trip + tamper-detection tests for the relay-mux wire format.
//!
//! Pure wire-layer tests: no sockets, no GPU. Records and length-framed
//! blobs are written into an in-memory buffer and read back, asserting
//! byte-exact round-trips and loud failures on corruption / salt
//! disagreement.

use super::*;

const SALT_A: SessionSalt = [
    0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10,
];
const SALT_B: SessionSalt = [
    0xa1, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xab, 0xac, 0xad, 0xae, 0xaf, 0xb0,
];

fn write_one(rec: &MuxRecord, salt: &SessionSalt) -> Vec<u8> {
    let mut buf = Vec::new();
    rec.write_to(&mut buf, salt).expect("write_to");
    buf
}

#[test]
fn data_record_round_trips() {
    let payload = vec![0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x11, 0x22];
    let rec = MuxRecord::data(7, payload.clone());
    let buf = write_one(&rec, &SALT_A);

    let mut cursor = &buf[..];
    let got = MuxRecord::read_from(&mut cursor, &SALT_A)
        .expect("read_from ok")
        .expect("not eof");
    assert_eq!(got, rec);
    match got {
        MuxRecord::Data { rank, payload: p } => {
            assert_eq!(rank, 7);
            assert_eq!(p, payload);
        }
        other => panic!("expected Data, got {other:?}"),
    }
    // Whole buffer consumed.
    assert!(cursor.is_empty(), "trailing bytes after record");
}

#[test]
fn empty_payload_data_record_round_trips() {
    let rec = MuxRecord::data(0, Vec::new());
    let buf = write_one(&rec, &SALT_A);
    let mut cursor = &buf[..];
    let got = MuxRecord::read_from(&mut cursor, &SALT_A).unwrap().unwrap();
    assert_eq!(got, rec);
}

#[test]
fn control_hello_round_trips() {
    let rec = MuxRecord::control(RelayControlMsg::Hello {
        host: "exa".into(),
        ranks: vec![0, 3, 4],
    });
    let buf = write_one(&rec, &SALT_A);
    let mut cursor = &buf[..];
    let got = MuxRecord::read_from(&mut cursor, &SALT_A).unwrap().unwrap();
    assert_eq!(got, rec);
}

#[test]
fn control_rank_exit_and_ack_round_trip() {
    for msg in [
        RelayControlMsg::RankExit { rank: 2 },
        RelayControlMsg::HelloAck,
        RelayControlMsg::DeclareDead { rank: 4 },
    ] {
        let rec = MuxRecord::control(msg.clone());
        let buf = write_one(&rec, &SALT_B);
        let mut cursor = &buf[..];
        let got = MuxRecord::read_from(&mut cursor, &SALT_B).unwrap().unwrap();
        assert_eq!(got, MuxRecord::Control(msg));
    }
}

#[test]
fn host_frame_and_broadcast_round_trip() {
    for rec in [
        MuxRecord::host_frame(vec![0xF0, 0x1D]),
        MuxRecord::broadcast(vec![0xCA, 0x5F]),
        MuxRecord::host_frame(Vec::new()),
        MuxRecord::broadcast(Vec::new()),
    ] {
        let buf = write_one(&rec, &SALT_A);
        let mut cursor = &buf[..];
        let got = MuxRecord::read_from(&mut cursor, &SALT_A).unwrap().unwrap();
        assert_eq!(got, rec);
        assert!(cursor.is_empty(), "trailing bytes after record");
    }
}

#[test]
fn host_frame_tampered_payload_fails_hmac() {
    let rec = MuxRecord::host_frame(vec![1, 2, 3, 4]);
    let mut buf = write_one(&rec, &SALT_A);
    let last = buf.len() - 1;
    buf[last] ^= 0xFF;
    let mut cursor = &buf[..];
    let err = MuxRecord::read_from(&mut cursor, &SALT_A).unwrap_err();
    assert!(err.to_string().contains("HMAC verification failed"), "got: {err}");
}

#[test]
fn multiple_records_stream_back_in_order() {
    // The relay writes many ranks' frames onto one connection; the
    // controller must read them back in write order, demuxing by tag.
    let recs = vec![
        MuxRecord::control(RelayControlMsg::Hello {
            host: "pascal".into(),
            ranks: vec![1, 2],
        }),
        MuxRecord::data(1, vec![0xAA; 5]),
        MuxRecord::data(2, vec![0xBB; 9]),
        MuxRecord::data(1, vec![0xCC; 1]),
        MuxRecord::control(RelayControlMsg::RankExit { rank: 2 }),
    ];
    let mut buf = Vec::new();
    for r in &recs {
        r.write_to(&mut buf, &SALT_A).unwrap();
    }
    let mut cursor = &buf[..];
    for expected in &recs {
        let got = MuxRecord::read_from(&mut cursor, &SALT_A).unwrap().unwrap();
        assert_eq!(&got, expected);
    }
    // Clean EOF after the last record.
    let mut empty = &buf[buf.len()..];
    assert!(MuxRecord::read_from(&mut empty, &SALT_A).unwrap().is_none());
}

#[test]
fn salt_disagreement_fails_hmac() {
    let rec = MuxRecord::data(3, vec![1, 2, 3, 4]);
    let buf = write_one(&rec, &SALT_A);
    let mut cursor = &buf[..];
    let err = MuxRecord::read_from(&mut cursor, &SALT_B).unwrap_err();
    assert!(
        err.to_string().contains("HMAC verification failed"),
        "expected HMAC error, got: {err}"
    );
}

#[test]
fn tampered_rank_tag_fails_hmac() {
    // Flipping the routing-sensitive rank field must be detected: the
    // mux header is authed, so a misroute attempt fails loudly.
    let rec = MuxRecord::data(5, vec![9, 9, 9]);
    let mut buf = write_one(&rec, &SALT_A);
    // rank lives at header bytes [9..13]; bump it.
    buf[9] = buf[9].wrapping_add(1);
    let mut cursor = &buf[..];
    let err = MuxRecord::read_from(&mut cursor, &SALT_A).unwrap_err();
    assert!(
        err.to_string().contains("HMAC verification failed"),
        "expected HMAC error on tampered rank, got: {err}"
    );
}

#[test]
fn tampered_payload_fails_hmac() {
    let rec = MuxRecord::data(0, vec![7, 7, 7, 7]);
    let mut buf = write_one(&rec, &SALT_A);
    *buf.last_mut().unwrap() ^= 0xFF;
    let mut cursor = &buf[..];
    let err = MuxRecord::read_from(&mut cursor, &SALT_A).unwrap_err();
    assert!(err.to_string().contains("HMAC verification failed"), "got: {err}");
}

#[test]
fn bad_magic_fails_loudly() {
    let rec = MuxRecord::data(0, vec![1]);
    let mut buf = write_one(&rec, &SALT_A);
    buf[0] ^= 0xFF;
    let mut cursor = &buf[..];
    let err = MuxRecord::read_from(&mut cursor, &SALT_A).unwrap_err();
    assert!(err.to_string().contains("magic"), "got: {err}");
}

#[test]
fn read_from_empty_is_clean_eof() {
    let buf: Vec<u8> = Vec::new();
    let mut cursor = &buf[..];
    assert!(MuxRecord::read_from(&mut cursor, &SALT_A).unwrap().is_none());
}

#[test]
fn try_read_from_reports_eof_on_empty() {
    let buf: Vec<u8> = Vec::new();
    let mut cursor = &buf[..];
    match MuxRecord::try_read_from(&mut cursor, &SALT_A).unwrap() {
        MuxRead::Eof => {}
        other => panic!("expected Eof, got {other:?}"),
    }
}

#[test]
fn try_read_from_decodes_record() {
    let rec = MuxRecord::data(11, vec![0x42; 3]);
    let buf = write_one(&rec, &SALT_A);
    let mut cursor = &buf[..];
    match MuxRecord::try_read_from(&mut cursor, &SALT_A).unwrap() {
        MuxRead::Record(got) => assert_eq!(got, rec),
        other => panic!("expected Record, got {other:?}"),
    }
}

// --- length-framed loopback leg ---

#[test]
fn len_framed_round_trips() {
    let blob = vec![0x10, 0x20, 0x30, 0x40, 0x50];
    let mut buf = Vec::new();
    write_len_framed(&mut buf, &blob).unwrap();
    // 4-byte prefix + body.
    assert_eq!(buf.len(), 4 + blob.len());
    let mut cursor = &buf[..];
    let got = read_len_framed(&mut cursor).unwrap().unwrap();
    assert_eq!(got, blob);
    assert!(cursor.is_empty());
}

#[test]
fn len_framed_empty_blob_round_trips() {
    let mut buf = Vec::new();
    write_len_framed(&mut buf, &[]).unwrap();
    let mut cursor = &buf[..];
    let got = read_len_framed(&mut cursor).unwrap().unwrap();
    assert!(got.is_empty());
}

#[test]
fn len_framed_multiple_in_sequence() {
    let blobs = [vec![1u8], vec![2, 2], vec![3, 3, 3], Vec::new(), vec![4; 100]];
    let mut buf = Vec::new();
    for b in &blobs {
        write_len_framed(&mut buf, b).unwrap();
    }
    let mut cursor = &buf[..];
    for expected in &blobs {
        let got = read_len_framed(&mut cursor).unwrap().unwrap();
        assert_eq!(&got, expected);
    }
    assert!(read_len_framed(&mut cursor).unwrap().is_none());
}

#[test]
fn len_framed_read_empty_is_clean_eof() {
    let buf: Vec<u8> = Vec::new();
    let mut cursor = &buf[..];
    assert!(read_len_framed(&mut cursor).unwrap().is_none());
}

#[test]
fn try_read_len_framed_reports_eof_and_blob() {
    let empty: Vec<u8> = Vec::new();
    let mut cursor = &empty[..];
    match try_read_len_framed(&mut cursor).unwrap() {
        LenFramedRead::Eof => {}
        other => panic!("expected Eof, got {other:?}"),
    }

    let blob = vec![5u8, 6, 7];
    let mut buf = Vec::new();
    write_len_framed(&mut buf, &blob).unwrap();
    let mut cursor = &buf[..];
    match try_read_len_framed(&mut cursor).unwrap() {
        LenFramedRead::Blob(got) => assert_eq!(got, blob),
        other => panic!("expected Blob, got {other:?}"),
    }
}

/// A writer that counts how many `write` calls it receives (accepting
/// every byte each call, so `write_all` issues exactly one `write` per
/// contiguous buffer).
struct CountingWriter {
    buf: Vec<u8>,
    writes: usize,
}

impl std::io::Write for CountingWriter {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.writes += 1;
        self.buf.extend_from_slice(b);
        Ok(b.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn record_is_written_atomically_single_write() {
    // Regression guard: a frame MUST be written in one `write_all` (one
    // buffer). Two writes (header then payload) open a mid-frame
    // preemption window where a reader on a timeout'd socket consumes a
    // partial frame and desyncs the stream — the bug that wedged the
    // cluster-coordinator CPU cycle tests. Both record kinds + the
    // length-framed loopback helper must each issue exactly one write.
    let mut w = CountingWriter { buf: Vec::new(), writes: 0 };
    MuxRecord::data(3, vec![1, 2, 3, 4, 5]).write_to(&mut w, &SALT_A).unwrap();
    assert_eq!(w.writes, 1, "Data record must be one atomic write");

    let mut w = CountingWriter { buf: Vec::new(), writes: 0 };
    MuxRecord::control(RelayControlMsg::Hello {
        host: "h".into(),
        ranks: vec![0, 1],
    })
    .write_to(&mut w, &SALT_A)
    .unwrap();
    assert_eq!(w.writes, 1, "Control record must be one atomic write");

    let mut w = CountingWriter { buf: Vec::new(), writes: 0 };
    write_len_framed(&mut w, &[9, 9, 9]).unwrap();
    assert_eq!(w.writes, 1, "len-framed blob must be one atomic write");
}

#[test]
fn opaque_payload_is_not_parsed() {
    // A Data record carries arbitrary bytes verbatim — the relay never
    // interprets them. Feed bytes that are NOT a valid RoundFrame /
    // ControlFrame and confirm they survive the round-trip untouched.
    let junk = vec![0xFF, 0x00, 0xFF, 0x00, 0xAB, 0xCD, 0xEF];
    let rec = MuxRecord::data(9, junk.clone());
    let buf = write_one(&rec, &SALT_A);
    let mut cursor = &buf[..];
    let got = MuxRecord::read_from(&mut cursor, &SALT_A).unwrap().unwrap();
    match got {
        MuxRecord::Data { payload, .. } => assert_eq!(payload, junk),
        other => panic!("expected Data, got {other:?}"),
    }
}

    // ---- payload ceiling ----------------------------------------------------

    /// The length fields are unauthenticated until the MAC verifies; a
    /// claimed length past the frame ceiling must be rejected before the
    /// reader commits to buffering it.
    #[test]
    fn oversized_len_framed_blob_is_rejected() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&u32::MAX.to_le_bytes());
        bytes.extend_from_slice(&[0u8; 16]); // some body bytes, never enough
        let mut cursor = std::io::Cursor::new(bytes);
        let err = try_read_len_framed(&mut cursor).unwrap_err();
        assert!(
            err.to_string().contains("frame ceiling"),
            "got: {err}"
        );
    }

    #[test]
    fn oversized_mux_record_is_rejected() {
        // Header: magic | version | kind | rank | payload_len | auth_tag.
        let mut hdr = Vec::new();
        hdr.extend_from_slice(&MUX_RECORD_MAGIC.to_le_bytes());
        hdr.extend_from_slice(&MUX_PROTOCOL_VERSION.to_le_bytes());
        hdr.push(0); // REC_DATA
        hdr.extend_from_slice(&0u32.to_le_bytes()); // rank
        hdr.extend_from_slice(&u32::MAX.to_le_bytes()); // hostile length
        hdr.extend_from_slice(&0u64.to_le_bytes()); // bogus tag
        let mut cursor = std::io::Cursor::new(hdr);
        let err = MuxRecord::read_from(&mut cursor, &SALT_A).unwrap_err();
        assert!(
            err.to_string().contains("frame ceiling"),
            "got: {err}"
        );
    }

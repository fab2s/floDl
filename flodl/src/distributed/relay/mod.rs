//! Per-host relay tier.
//!
//! A relay agent runs once per host and collapses the cluster's
//! one-TCP-connection-per-rank topology into one connection per host per
//! channel. Toward its local ranks it presents the existing controller
//! protocol over loopback; upstream toward the real controller it speaks
//! the muxed protocol in [`mux`]. On the control channel it forwards,
//! tagging each frame with its originating rank; on the data channel it
//! is the first fold tier — local `RoundFrame`s are summed into ONE
//! `HostFrame` per round (see [`agent`]), so uplink bytes scale per host,
//! not per rank.
//!
//! See [`mux`] for the wire format and [`agent`] for the byte-router +
//! fold station.
//!
//! # Port map
//!
//! The controller process accepts every cross-host channel — NCCL
//! rendezvous, CPU-reduce data, coordinator control — on ONE port
//! (`controller.port`, bound `0.0.0.0`), demuxed by each connection's
//! channel-select magic (see `port_mux`).
//!
//! Each host's relay binds two LOOPBACK ports its local ranks dial, and
//! forwards upstream to the controller's mux port:
//!
//! ```text
//! 127.0.0.1:+4  relay data    loopback  → controller.host:port
//! 127.0.0.1:+5  relay control loopback  → controller.host:port
//! ```
//!
//! The `+4`/`+5` loopback ports never collide with the controller's
//! `0.0.0.0:port` even when the relay shares the controller's host.

pub mod agent;
pub mod mux;

/// Loopback port offset (from `controller.port`) the per-host relay binds
/// for the CPU-averaging data channel.
pub const RELAY_DATA_LOOPBACK_OFFSET: u16 = 4;

/// Loopback port offset (from `controller.port`) the per-host relay binds
/// for the coordinator control channel.
pub const RELAY_CONTROL_LOOPBACK_OFFSET: u16 = 5;

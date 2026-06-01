//! Per-host relay transport tier (v1: transport only).
//!
//! A relay agent runs once per host and collapses the cluster's
//! one-TCP-connection-per-rank topology into one connection per host per
//! channel. Toward its local ranks it presents the existing controller
//! protocol over loopback; upstream toward the real controller it speaks
//! the muxed protocol in [`mux`], tagging each forwarded frame with its
//! originating rank.
//!
//! The relay FORWARDS — it does not aggregate. One-rank-one-process,
//! averaging math, and the snapshot path are all unchanged; this tier is
//! purely connection topology + frame routing. The N×-fewer-bytes wire
//! reduction (sum-and-count) is a separate later layer that will live
//! here.
//!
//! See [`mux`] for the wire format and [`agent`] for the byte-router.
//!
//! # Port map
//!
//! The controller process owns these ports (all bound `0.0.0.0`), keyed
//! off `controller.port` (the base):
//!
//! ```text
//! +0  NCCL rendezvous (bootstrap; stays direct — not relayed in v1)
//! +1  reserved (dashboard side-channel)
//! +2  ClusterController  (CPU-averaging data channel)
//! +3  ClusterCoordinator (control channel)
//! ```
//!
//! Each host's relay binds two LOOPBACK ports its local ranks dial, and
//! forwards upstream to the controller's `+2` / `+3`:
//!
//! ```text
//! 127.0.0.1:+4  relay data    loopback  → controller.host:+2
//! 127.0.0.1:+5  relay control loopback  → controller.host:+3
//! ```
//!
//! The `+4`/`+5` loopback ports never collide with the controller's
//! `0.0.0.0:+2/+3` even when the relay shares the controller's host.

pub mod agent;
pub mod mux;

/// Loopback port offset (from `controller.port`) the per-host relay binds
/// for the CPU-averaging data channel; forwards upstream to `+2`.
pub const RELAY_DATA_LOOPBACK_OFFSET: u16 = 4;

/// Loopback port offset (from `controller.port`) the per-host relay binds
/// for the coordinator control channel; forwards upstream to `+3`.
pub const RELAY_CONTROL_LOOPBACK_OFFSET: u16 = 5;

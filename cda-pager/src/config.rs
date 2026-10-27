/*
 * SPDX-FileCopyrightText: 2026 Copyright (c) Contributors to the Eclipse Foundation
 *
 * See the NOTICE file(s) distributed with this work for additional
 * information regarding copyright ownership.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * SPDX-License-Identifier: Apache-2.0
 */

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::replacement::EvictionKind;

/// Default speculative read-ahead per fault, in bytes. Converted to pages
/// via the configured page size (e.g. 32 pages at 4 KiB).
pub const DEFAULT_PREFETCH_BYTES: usize = 128 * 1024;

const DEFAULT_PAGE_SIZE: usize = 4096;

/// Tunables for the global pager. All values are process-wide: a single hub
/// serves every opened MDD file and `pool_size_bytes` caps the number of
/// resident pages across all of them combined.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, JsonSchema)]
#[serde(default)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "Configuration struct legitimately has multiple independent boolean fields"
)]
pub struct PagerConfig {
    /// Load MDD files through the userfaultfd based pager instead of plain
    /// `mmap`. Linux only; enabling it on other platforms is an error.
    pub enabled: bool,
    /// Global budget for resident pages across all open MDD files, in bytes.
    pub pool_size_bytes: usize,
    /// Pager page size in bytes. Must be a power of two and at least 4096.
    pub page_size: usize,
    /// Number of contiguous pages speculatively loaded per fault.
    pub prefetch_pages: usize,
    /// Load the prefetch batch preceding the faulting page instead of the
    /// pages following it.
    pub prefetch_backward: bool,
    /// Page replacement strategy used to pick eviction victims.
    pub eviction: EvictionKind,
    /// Seed for policies with a random component.
    pub eviction_seed: u64,
    /// Minimum fraction (0.0..=1.0) of the pool to evict whenever eviction
    /// is triggered. Batching amortizes the eviction overhead.
    pub eviction_batch_fraction: f64,
    /// Read file data with `O_DIRECT`, bypassing the kernel page cache.
    /// Falls back to buffered reads if the filesystem rejects it.
    pub direct_io: bool,
}

impl Default for PagerConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            pool_size_bytes: 64 * 1024 * 1024,
            page_size: DEFAULT_PAGE_SIZE,
            prefetch_pages: DEFAULT_PREFETCH_BYTES
                .checked_div(DEFAULT_PAGE_SIZE)
                .unwrap_or(1),
            prefetch_backward: false,
            eviction: EvictionKind::default(),
            eviction_seed: 42,
            eviction_batch_fraction: 0.0,
            direct_io: false,
        }
    }
}

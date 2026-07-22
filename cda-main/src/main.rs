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

#[tokio::main]
async fn main() -> Result<(), opensovd_cda_lib::AppError> {
    // Keep the profiler alive for the whole process lifetime; on shutdown its
    // Drop impl writes `dhat-heap.json`.
    #[cfg(feature = "dhat-heap")]
    let _dhat_profiler = opensovd_cda_lib::init_heap_profiler();
    opensovd_cda_lib::run_from_cli().await
}

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

//! Pluggable page-replacement strategies for the userfaultfd pager.
//!
//! # What a policy can and cannot observe
//!
//! The pager registers its arenas with userfaultfd in MISSING-only mode, so
//! at the kernel level we only ever learn about *non-resident* pages: the
//! **miss stream**. There is no hit stream: a page that is already resident
//! is read straight from memory, with no kernel event and no instrumentation
//! on the `FlatBuffers` read path, so [`ReplacementPolicy::on_hit`] is never
//! called in this build.
//!
//! - Without hits, recency-ordered policies ([`LruHashMap`] and friends,
//!   [`TwoQueue`]'s protected queue, [`Clock`]'s reference bits) can only
//!   order pages by their fault history, not by re-touches. They remain
//!   available, but their rankings degenerate towards insertion order.
//! - Admission is a separate question from retention: whether a freshly
//!   re-faulted page was evicted too soon is answered by a **ghost list**
//!   (remember recently-evicted identities, treat a fast re-fault on one as
//!   evidence of a bad eviction), gated on [`ReplacementPolicy::on_fault`].
//!
//! Similarly, [`ReplacementPolicy::on_insert`] reports whether a page was
//! demand-faulted or merely speculatively prefetched alongside it, but a
//! prefetched page can never be *ghost-hit promoted*, because prefetching
//! never calls `on_fault` for it. See [`PrefetchFirst`] for a policy with no
//! recovery path from that at all.
//!
//! # Ownership contract
//!
//! [`ReplacementPolicy::take_victims`] *removes* the pages it returns from
//! the policy's own bookkeeping. A policy must never return a page again
//! until a matching [`ReplacementPolicy::on_insert`]. This keeps each
//! policy's notion of residency exact by construction, so the pager needs no
//! stale-entry filtering and no residency back-channel into the policy.
//!
//! This module is deliberately free of `unsafe`, `libc` and `userfaultfd`,
//! so it is unit-testable on any platform without kernel support.

#![allow(
    clippy::cast_possible_truncation,
    reason = "Slot indices are `u32` and bounded by the resident page pool size"
)]
#![allow(
    clippy::arithmetic_side_effects,
    reason = "Page index and counter arithmetic on bounded values in the paging hot path; \
              overflow is impossible by construction (indices are below the region page count)"
)]
#![allow(
    clippy::indexing_slicing,
    reason = "Indices are bounded by the region page count / batch size established when the \
              region is opened, a failed lookup would indicate a broken pager invariant"
)]

use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

/// Identifies a region (one opened file). Equal to the region's arena base
/// address, mirroring the key used by the pager's region map.
pub type RegionId = usize;

/// A single page of a single region: the unit of replacement.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct PageRef {
    pub region: RegionId,
    pub index: usize,
}

impl PageRef {
    #[must_use]
    pub fn new(region: RegionId, index: usize) -> Self {
        Self { region, index }
    }
}

/// A page-replacement strategy. See the module docs for what a policy can
/// observe and for the ownership contract of [`Self::take_victims`].
pub trait ReplacementPolicy: Send {
    /// Stable identifier, emitted into the metrics CSV so benchmark runs can
    /// be grouped by policy.
    fn name(&self) -> &'static str;

    /// A resident page was touched again (a logical hit, not a fault).
    /// Never called in this build — resident-page reads are invisible to
    /// the pager (see the module docs). Default no-op; only policies with a
    /// genuine recency signal act on this.
    #[allow(
        dead_code,
        reason = "Hook for policies with a hit signal, not driven by the pager"
    )]
    fn on_hit(&mut self, _page: PageRef) {}

    /// A miss was observed on `page`, called *before* the corresponding
    /// [`Self::on_insert`]. This is the only recency signal available;
    /// ghost-list policies key off it.
    fn on_fault(&mut self, page: PageRef);

    /// `page` has become resident. `demand` is true only for the page that
    /// actually faulted, false for pages pulled in by the prefetch batch
    /// around it.
    fn on_insert(&mut self, page: PageRef, demand: bool);

    /// Remove up to `need` pages from the policy's resident set and append
    /// them to `out`. Returning fewer than `need` means "nothing more to
    /// give" and stops the pager's eviction loop.
    ///
    /// Order does not matter: the pager sorts the batch before issuing
    /// `madvise`, so a policy is free to return scattered pages without
    /// having to think about syscall coalescing.
    fn take_victims(&mut self, need: usize, out: &mut Vec<PageRef>);

    /// Every page of `region` is gone (the file handle was dropped). All
    /// state referring to it must be purged.
    fn on_region_closed(&mut self, region: RegionId);

    /// A region was opened. `referenced` holds one reference bit per page,
    /// set lock-free by the `FlatBuffers` read path when
    /// [`EvictionKind::uses_ref_bits`] is true. Default no-op; only
    /// [`Clock`] uses it.
    fn on_region_opened(&mut self, _region: RegionId, _referenced: Arc<[AtomicBool]>) {}

    /// Number of faults that hit a ghost-list entry, i.e. pages this policy
    /// evicted and then immediately needed again. Zero for policies without
    /// a ghost list.
    fn ghost_hits(&self) -> u64 {
        0
    }
}

/// Selects which [`ReplacementPolicy`] implementation to instantiate.
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
    schemars::JsonSchema,
)]
#[serde(rename_all = "kebab-case")]
pub enum EvictionKind {
    /// Evict the longest-resident page. The pager's original behaviour.
    Fifo,
    Random,
    PrefetchFirst,
    TwoQueue,
    LruHashMap,
    /// Same eviction order as [`Self::LruHashMap`], slot-indexed instead of
    /// hashed. See [`LruLinkedList`]. The default: O(1) hit and eviction.
    #[default]
    LruLinkedList,
    /// Same eviction order as [`Self::LruHashMap`], recency stamped on hit
    /// and resolved at eviction time. See [`LruCounter`].
    LruCounter,
    /// Second-chance / CLOCK: LRU approximation driven by lock-free
    /// per-page reference bits instead of per-hit bookkeeping. See [`Clock`].
    Clock,
}

impl EvictionKind {
    /// Accepted spellings for the `--pager-eviction` flag.
    pub const CHOICES: &'static str =
        "fifo|random|prefetch-first|twoq|lru-hashmap|lru-linked-list|lru-counter|clock";

    /// Parses one of the spellings listed in [`Self::CHOICES`].
    ///
    /// # Errors
    /// Returns an error naming the accepted values if `s` is unknown.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "fifo" => Ok(Self::Fifo),
            "random" | "rand" => Ok(Self::Random),
            "prefetch-first" | "prefetch" => Ok(Self::PrefetchFirst),
            "twoq" | "2q" => Ok(Self::TwoQueue),
            "lru-hashmap" | "lru_hashmap" => Ok(Self::LruHashMap),
            "lru-linked-list" | "lru_linked_list" | "lru-list" => Ok(Self::LruLinkedList),
            "lru-counter" | "lru_counter" => Ok(Self::LruCounter),
            "clock" | "second-chance" => Ok(Self::Clock),
            other => Err(format!(
                "unknown eviction policy '{other}', expected one of: {}",
                Self::CHOICES
            )),
        }
    }

    /// Whether the policy reads per-page reference bits. The bits are
    /// allocated per region only for such policies.
    #[must_use]
    pub fn uses_ref_bits(&self) -> bool {
        matches!(self, Self::Clock)
    }

    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Fifo => "fifo",
            Self::Random => "random",
            Self::PrefetchFirst => "prefetch-first",
            Self::TwoQueue => "twoq",
            Self::LruHashMap => "lru-hashmap",
            Self::LruLinkedList => "lru-linked-list",
            Self::LruCounter => "lru-counter",
            Self::Clock => "clock",
        }
    }
}

/// Instantiates a policy. `seed` makes [`Random`] reproducible across runs
/// (wired to the simulator's `--seed`); `capacity_hint` is the hub's global
/// resident-page cap, used to size ghost lists.
#[must_use]
pub fn build(kind: EvictionKind, seed: u64, capacity_hint: usize) -> Box<dyn ReplacementPolicy> {
    match kind {
        EvictionKind::Fifo => Box::new(Fifo::new()),
        EvictionKind::Random => Box::new(Random::new(seed)),
        EvictionKind::PrefetchFirst => Box::new(PrefetchFirst::new()),
        EvictionKind::TwoQueue => Box::new(TwoQueue::new(capacity_hint)),
        EvictionKind::LruHashMap => Box::new(LruHashMap::new()),
        EvictionKind::LruLinkedList => Box::new(LruLinkedList::with_capacity(capacity_hint)),
        EvictionKind::LruCounter => Box::new(LruCounter::with_capacity(capacity_hint)),
        EvictionKind::Clock => Box::new(Clock::new()),
    }
}

/// Drops every entry belonging to `region`. O(n), but region closes are rare
/// (once per opened file) and this keeps the queues free of stale entries,
/// which is what lets `take_victims` guarantee it only returns resident
/// pages.
fn purge_region(queue: &mut VecDeque<PageRef>, region: RegionId) {
    queue.retain(|page| page.region != region);
}

/// Evicts the page that has been resident longest, regardless of which
/// region it belongs to. The pager's original (and default) behaviour.
pub struct Fifo {
    queue: VecDeque<PageRef>,
}

impl Fifo {
    #[must_use]
    pub fn new() -> Self {
        Self {
            queue: VecDeque::new(),
        }
    }
}

impl Default for Fifo {
    fn default() -> Self {
        Self::new()
    }
}

impl ReplacementPolicy for Fifo {
    fn name(&self) -> &'static str {
        "fifo"
    }

    fn on_fault(&mut self, _page: PageRef) {}

    fn on_insert(&mut self, page: PageRef, _demand: bool) {
        self.queue.push_back(page);
    }

    fn take_victims(&mut self, need: usize, out: &mut Vec<PageRef>) {
        for _ in 0..need {
            match self.queue.pop_front() {
                Some(page) => out.push(page),
                None => break,
            }
        }
    }

    fn on_region_closed(&mut self, region: RegionId) {
        purge_region(&mut self.queue, region);
    }
}

/// Second-chance / CLOCK replacement.
///
/// Resident pages sit in a ring in insertion order; the front is the clock
/// hand. Recency comes from per-page *reference bits* that the `FlatBuffers`
/// read path sets with a single relaxed atomic store (no lock, no
/// per-hit bookkeeping, unlike the exact LRU variants). On eviction the hand
/// takes the front page: if its bit is set, the bit is cleared and the page
/// is moved to the back (second chance); otherwise it is evicted. Each page
/// survives at most one pass of the hand, so a call terminates after at most
/// one full rotation plus `need` steps.
///
/// Newly inserted pages start with their bit set only if they were demand
/// faulted: prefetched neighbours must earn their second chance by actually
/// being read.
pub struct Clock {
    ring: VecDeque<PageRef>,
    referenced: HashMap<RegionId, Arc<[AtomicBool]>>,
}

impl Clock {
    #[must_use]
    pub fn new() -> Self {
        Self {
            ring: VecDeque::new(),
            referenced: HashMap::new(),
        }
    }

    fn bit(&self, page: PageRef) -> Option<&AtomicBool> {
        self.referenced.get(&page.region)?.get(page.index)
    }
}

impl Default for Clock {
    fn default() -> Self {
        Self::new()
    }
}

impl ReplacementPolicy for Clock {
    fn name(&self) -> &'static str {
        "clock"
    }

    fn on_fault(&mut self, _page: PageRef) {}

    fn on_insert(&mut self, page: PageRef, demand: bool) {
        if let Some(bit) = self.bit(page) {
            bit.store(demand, Ordering::Relaxed);
        }
        self.ring.push_back(page);
    }

    fn take_victims(&mut self, need: usize, out: &mut Vec<PageRef>) {
        let mut taken = 0;
        // Upper bound on second chances: one full rotation.
        let mut spins = self.ring.len();
        while taken < need {
            let Some(page) = self.ring.pop_front() else {
                break;
            };
            if spins > 0
                && self
                    .bit(page)
                    .is_some_and(|b| b.swap(false, Ordering::Relaxed))
            {
                spins -= 1;
                self.ring.push_back(page);
                continue;
            }
            out.push(page);
            taken += 1;
        }
    }

    fn on_region_opened(&mut self, region: RegionId, referenced: Arc<[AtomicBool]>) {
        self.referenced.insert(region, referenced);
    }

    fn on_region_closed(&mut self, region: RegionId) {
        purge_region(&mut self.ring, region);
        self.referenced.remove(&region);
    }
}

/// Minimal seedable PRNG (`SplitMix64`); good enough for victim selection and
/// avoids pulling in a random number crate.
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform-enough value in `0..bound`; `bound` must be non-zero.
    fn below(&mut self, bound: usize) -> usize {
        usize::try_from(self.next_u64() % (bound as u64).max(1)).unwrap_or(0)
    }
}

/// Uniformly random victim selection, the standard baseline that any
/// smarter policy has to beat. Seeded for reproducibility.
///
/// Kept O(1) per operation by pairing a dense `Vec` with an index map, so
/// removal is a `swap_remove` rather than a linear scan.
pub struct Random {
    pages: Vec<PageRef>,
    index_of: HashMap<PageRef, usize>,
    rng: SplitMix64,
}

impl Random {
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            pages: Vec::new(),
            index_of: HashMap::new(),
            rng: SplitMix64::new(seed),
        }
    }

    fn remove_at(&mut self, slot: usize) -> PageRef {
        let page = self.pages.swap_remove(slot);
        self.index_of.remove(&page);
        // `swap_remove` moved the former last element into `slot`, unless we
        // removed the last element itself.
        if let Some(moved) = self.pages.get(slot) {
            self.index_of.insert(*moved, slot);
        }
        page
    }
}

impl ReplacementPolicy for Random {
    fn name(&self) -> &'static str {
        "random"
    }

    fn on_fault(&mut self, _page: PageRef) {}

    fn on_insert(&mut self, page: PageRef, _demand: bool) {
        if self.index_of.contains_key(&page) {
            return;
        }
        self.index_of.insert(page, self.pages.len());
        self.pages.push(page);
    }

    fn take_victims(&mut self, need: usize, out: &mut Vec<PageRef>) {
        for _ in 0..need {
            if self.pages.is_empty() {
                break;
            }
            let slot = self.rng.below(self.pages.len());
            let page = self.remove_at(slot);
            out.push(page);
        }
    }

    fn on_region_closed(&mut self, region: RegionId) {
        self.pages.retain(|page| page.region != region);
        self.index_of.clear();
        for (slot, page) in self.pages.iter().enumerate() {
            self.index_of.insert(*page, slot);
        }
    }
}

/// Evicts speculatively prefetched pages before demand-faulted ones, on the
/// theory that a demand fault is proof the page was actually wanted whereas
/// a prefetched neighbour is only a guess.
///
/// Caveat, and it is a real one: this policy ignores
/// [`ReplacementPolicy::on_hit`] entirely, so a prefetched page that turns
/// out to be *heavily* used can never be promoted out of the speculative
/// queue. It stays a first-choice victim for as long as it is resident. Its
/// only other possible corrective feedback is the re-fault it causes after
/// being evicted, which this policy does not act on either — see
/// [`TwoQueue`] for a policy that uses both signals.
pub struct PrefetchFirst {
    speculative: VecDeque<PageRef>,
    demand: VecDeque<PageRef>,
}

impl PrefetchFirst {
    #[must_use]
    pub fn new() -> Self {
        Self {
            speculative: VecDeque::new(),
            demand: VecDeque::new(),
        }
    }
}

impl Default for PrefetchFirst {
    fn default() -> Self {
        Self::new()
    }
}

impl ReplacementPolicy for PrefetchFirst {
    fn name(&self) -> &'static str {
        "prefetch-first"
    }

    fn on_fault(&mut self, _page: PageRef) {}

    fn on_insert(&mut self, page: PageRef, demand: bool) {
        if demand {
            self.demand.push_back(page);
        } else {
            self.speculative.push_back(page);
        }
    }

    fn take_victims(&mut self, need: usize, out: &mut Vec<PageRef>) {
        for _ in 0..need {
            let picked = self
                .speculative
                .pop_front()
                .or_else(|| self.demand.pop_front());
            match picked {
                Some(page) => out.push(page),
                None => break,
            }
        }
    }

    fn on_region_closed(&mut self, region: RegionId) {
        purge_region(&mut self.speculative, region);
        purge_region(&mut self.demand, region);
    }
}

/// A 2Q-style policy combining two recency signals: a ghost list decides
/// which pages are worth *admitting* into `protected`, and a CLOCK-style
/// reference bit (set via [`ReplacementPolicy::on_hit`]) decides which
/// `protected` pages are worth *keeping* once eviction reaches that far.
///
/// New pages enter a probationary FIFO (`probation`). When a page is
/// evicted its identity (not its data) is remembered in a bounded ghost
/// list. If that page faults again while still in the ghost list, the
/// eviction was premature, so on re-admission it goes into `protected`
/// instead, which is only drained once `probation` is empty. Pages that are
/// touched once and never again therefore cycle through probation without
/// ever displacing the working set.
///
/// Within `protected`, evicting in plain FIFO order would ignore any reuse
/// that happens after promotion. Instead each resident page there carries a
/// reference bit, set by `on_hit`; eviction sweeps `protected` in FIFO order
/// but gives a referenced page one second chance (bit cleared, requeued to
/// the back) before actually evicting it — see `clock_evict`.
pub struct TwoQueue {
    probation: VecDeque<PageRef>,
    protected: VecDeque<PageRef>,
    protected_capacity: usize,
    protected_set: HashSet<PageRef>,
    referenced: HashSet<PageRef>,
    ghost: VecDeque<PageRef>,
    ghost_set: HashSet<PageRef>,
    ghost_capacity: usize,
    /// Set by `on_fault` when the faulting page was a ghost hit, consumed by
    /// the matching `on_insert`. Only ever holds the demand-faulted page,
    /// never a prefetched neighbour.
    promote: Option<PageRef>,
    stat_ghost_hits: u64,
}

impl TwoQueue {
    #[must_use]
    pub fn new(capacity_hint: usize) -> Self {
        // `protected` gets the full capacity hint, since CLOCK's
        // second-chance requeuing needs room to actually distinguish
        // referenced pages from unreferenced ones; `ghost` only needs to
        // cover about half a cache turnover of evicted identities to catch
        // premature evictions.
        let ghost_capacity = capacity_hint.div_ceil(2).max(1);
        Self {
            probation: VecDeque::new(),
            protected: VecDeque::new(),
            protected_capacity: capacity_hint.max(1),
            protected_set: HashSet::new(),
            referenced: HashSet::new(),
            ghost: VecDeque::with_capacity(ghost_capacity),
            ghost_set: HashSet::new(),
            ghost_capacity,
            promote: None,
            stat_ghost_hits: 0,
        }
    }

    fn remember_evicted(&mut self, page: PageRef) {
        if self.ghost_set.insert(page) {
            self.ghost.push_back(page);
        }
        while self.ghost.len() > self.ghost_capacity {
            if let Some(oldest) = self.ghost.pop_front() {
                self.ghost_set.remove(&oldest);
            }
        }
    }

    /// One CLOCK sweep over `protected`: a referenced page gets a second
    /// chance (bit cleared, requeued to the back); the first unreferenced
    /// page found is removed and returned.
    fn clock_evict(&mut self) -> Option<PageRef> {
        loop {
            let page = self.protected.pop_front()?;
            if self.referenced.remove(&page) {
                self.protected.push_back(page);
            } else {
                self.protected_set.remove(&page);
                return Some(page);
            }
        }
    }
}

impl ReplacementPolicy for TwoQueue {
    fn name(&self) -> &'static str {
        "twoq"
    }

    fn on_hit(&mut self, page: PageRef) {
        if self.protected_set.contains(&page) {
            self.referenced.insert(page);
        }
    }

    fn on_fault(&mut self, page: PageRef) {
        if self.ghost_set.remove(&page) {
            self.stat_ghost_hits += 1;
            self.promote = Some(page);
        }
    }

    fn on_insert(&mut self, page: PageRef, demand: bool) {
        let promote = demand && self.promote == Some(page);
        if promote {
            self.promote = None;
            self.protected.push_back(page);
            self.protected_set.insert(page);
            if self.protected.len() > self.protected_capacity {
                if let Some(demoted) = self.clock_evict() {
                    self.probation.push_back(demoted);
                }
            }
        } else {
            self.probation.push_back(page);
        }
    }

    fn take_victims(&mut self, need: usize, out: &mut Vec<PageRef>) {
        for _ in 0..need {
            let picked = self.probation.pop_front().or_else(|| self.clock_evict());
            match picked {
                Some(page) => {
                    self.remember_evicted(page);
                    out.push(page);
                }
                None => break,
            }
        }
    }

    fn on_region_closed(&mut self, region: RegionId) {
        purge_region(&mut self.probation, region);
        purge_region(&mut self.protected, region);
        purge_region(&mut self.ghost, region);
        self.ghost_set.retain(|page| page.region != region);
        self.protected_set.retain(|page| page.region != region);
        self.referenced.retain(|page| page.region != region);
        if matches!(self.promote, Some(page) if page.region == region) {
            self.promote = None;
        }
    }

    fn ghost_hits(&self) -> u64 {
        self.stat_ghost_hits
    }
}

/// One node of [`LruHashMap`]'s intrusive doubly-linked list, keyed by
/// `PageRef` in `LruHashMap::nodes` rather than stored inline, so moving a
/// page to the most-recently-used end never invalidates another page's
/// position.
struct LruHashMapNode {
    prev: Option<PageRef>,
    next: Option<PageRef>,
}

/// Textbook recency-ordered LRU: [`ReplacementPolicy::on_hit`] moves the
/// touched page to the most-recently-used end, and eviction always takes
/// the least-recently-used end. This is only well-founded because
/// `TracedBuf` (see module docs) reports every resident-page re-touch, not
/// just faults — without that hit stream, userfaultfd alone cannot
/// distinguish a hot resident page from a cold one.
///
/// Implemented as an intrusive doubly-linked list over a `HashMap` so that
/// `on_hit` (move to back), `on_insert` (push back) and `take_victims`
/// (pop front) are all O(1), rather than the O(n) a `VecDeque`-based
/// move-to-back would need.
pub struct LruHashMap {
    nodes: HashMap<PageRef, LruHashMapNode>,
    head: Option<PageRef>, // least recently used
    tail: Option<PageRef>, // most recently used
}

impl LruHashMap {
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            head: None,
            tail: None,
        }
    }

    /// Unlinks `page` from the list without removing its node entry.
    /// Callers must either re-link it (`push_back`) or remove the node
    /// themselves (`pop_front`).
    fn unlink(&mut self, page: PageRef) {
        let (prev, next) = {
            let node = self.nodes.get(&page).expect("unlink of untracked page");
            (node.prev, node.next)
        };
        if let Some(p) = prev {
            if let Some(node) = self.nodes.get_mut(&p) {
                node.next = next;
            }
        } else {
            self.head = next;
        }
        if let Some(n) = next {
            if let Some(node) = self.nodes.get_mut(&n) {
                node.prev = prev;
            }
        } else {
            self.tail = prev;
        }
    }

    fn push_back(&mut self, page: PageRef) {
        let old_tail = self.tail;
        self.nodes.insert(
            page,
            LruHashMapNode {
                prev: old_tail,
                next: None,
            },
        );
        if let Some(t) = old_tail {
            if let Some(node) = self.nodes.get_mut(&t) {
                node.next = Some(page);
            }
        } else {
            self.head = Some(page);
        }
        self.tail = Some(page);
    }

    fn pop_front(&mut self) -> Option<PageRef> {
        let page = self.head?;
        self.unlink(page);
        self.nodes.remove(&page);
        Some(page)
    }
}

impl Default for LruHashMap {
    fn default() -> Self {
        Self::new()
    }
}

impl ReplacementPolicy for LruHashMap {
    fn name(&self) -> &'static str {
        "lru-hashmap"
    }

    fn on_hit(&mut self, page: PageRef) {
        // A hit on a page not yet tracked (e.g. one that raced with
        // eviction) is impossible by the ownership contract, but ignore it
        // defensively rather than panicking.
        if self.nodes.contains_key(&page) {
            self.unlink(page);
            self.push_back(page);
        }
    }

    fn on_fault(&mut self, _page: PageRef) {}

    fn on_insert(&mut self, page: PageRef, _demand: bool) {
        self.push_back(page);
    }

    fn take_victims(&mut self, need: usize, out: &mut Vec<PageRef>) {
        for _ in 0..need {
            match self.pop_front() {
                Some(page) => out.push(page),
                None => break,
            }
        }
    }

    fn on_region_closed(&mut self, region: RegionId) {
        let stale: Vec<PageRef> = self
            .nodes
            .keys()
            .filter(|page| page.region == region)
            .copied()
            .collect();
        for page in stale {
            self.unlink(page);
            self.nodes.remove(&page);
        }
    }
}

/// Sentinel for "this page has no slot", and for the null link at either
/// end of [`LruLinkedList`].
const NO_SLOT: u32 = u32::MAX;

/// Maps a [`PageRef`] to a dense `u32` slot without hashing.
///
/// Both slot-indexed LRU variants below need `PageRef -> slot` on every
/// hit. Regions are few — one per open file — so a linear scan with a
/// most-recently-used hint beats hashing a 16-byte key, and the per-region
/// `Vec<u32>` reduces the page lookup to plain array indexing.
struct SlotTable {
    regions: Vec<(RegionId, Vec<u32>)>,
    hint: usize,
}

impl SlotTable {
    fn new() -> Self {
        Self {
            regions: Vec::new(),
            hint: 0,
        }
    }

    /// Position of `region` in `self.regions`, appending an empty entry if
    /// this is the first time we see it.
    fn region_pos(&mut self, region: RegionId) -> usize {
        if matches!(self.regions.get(self.hint), Some(&(id, _)) if id == region) {
            return self.hint;
        }
        let pos = if let Some(pos) = self.regions.iter().position(|(id, _)| *id == region) {
            pos
        } else {
            self.regions.push((region, Vec::new()));
            self.regions.len() - 1
        };
        self.hint = pos;
        pos
    }

    fn get(&mut self, page: PageRef) -> u32 {
        let pos = self.region_pos(page.region);
        self.regions[pos]
            .1
            .get(page.index)
            .copied()
            .unwrap_or(NO_SLOT)
    }

    fn set(&mut self, page: PageRef, slot: u32) {
        let pos = self.region_pos(page.region);
        let slots = &mut self.regions[pos].1;
        if page.index >= slots.len() {
            slots.resize(page.index + 1, NO_SLOT);
        }
        slots[page.index] = slot;
    }

    fn clear(&mut self, page: PageRef) {
        let pos = self.region_pos(page.region);
        if let Some(slot) = self.regions[pos].1.get_mut(page.index) {
            *slot = NO_SLOT;
        }
    }

    /// Forgets `region` and hands back its slot vector so the caller can
    /// release whatever those slots referred to.
    fn remove_region(&mut self, region: RegionId) -> Vec<u32> {
        self.hint = 0;
        match self.regions.iter().position(|(id, _)| *id == region) {
            Some(pos) => self.regions.swap_remove(pos).1,
            None => Vec::new(),
        }
    }
}

struct LruListNode {
    page: PageRef,
    prev: u32,
    next: u32,
}

/// Exactly [`LruHashMap`]'s eviction order, but with the
/// `HashMap<PageRef, _>` node arena replaced by a packed `Vec` addressed
/// through a [`SlotTable`].
///
/// `LruHashMap` pays roughly five hashes of a 16-byte key per `on_hit`
/// (read the
/// node, fix both neighbours, re-insert, fix the old tail), each landing at
/// an unpredictable bucket. Here every one of those becomes a direct index
/// into a contiguous `Vec`, so the hit path does no hashing, no allocation
/// and no rehash-triggered resize, and the nodes it walks are far more
/// likely to already be in cache.
///
/// Eviction decisions are identical to `LruHashMap` page for page, so any
/// difference in the metrics CSV beyond `duration_us` / `cpu_*_us` is a bug
/// in one of the two.
pub struct LruLinkedList {
    nodes: Vec<LruListNode>,
    free: Vec<u32>,
    slots: SlotTable,
    head: u32, // least recently used
    tail: u32, // most recently used
}

impl LruLinkedList {
    #[must_use]
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    /// `capacity` is the hub's resident-page cap: the list never holds more
    /// nodes than that, so reserving up front removes every regrow.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(capacity),
            free: Vec::new(),
            slots: SlotTable::new(),
            head: NO_SLOT,
            tail: NO_SLOT,
        }
    }

    fn unlink(&mut self, slot: u32) {
        let (prev, next) = {
            let node = &self.nodes[slot as usize];
            (node.prev, node.next)
        };
        if prev == NO_SLOT {
            self.head = next;
        } else {
            self.nodes[prev as usize].next = next;
        }
        if next == NO_SLOT {
            self.tail = prev;
        } else {
            self.nodes[next as usize].prev = prev;
        }
    }

    /// Re-links an already-allocated `slot` at the most-recently-used end.
    fn link_back(&mut self, slot: u32) {
        let old_tail = self.tail;
        {
            let node = &mut self.nodes[slot as usize];
            node.prev = old_tail;
            node.next = NO_SLOT;
        }
        if old_tail == NO_SLOT {
            self.head = slot;
        } else {
            self.nodes[old_tail as usize].next = slot;
        }
        self.tail = slot;
    }

    fn alloc(&mut self, page: PageRef) -> u32 {
        let node = LruListNode {
            page,
            prev: NO_SLOT,
            next: NO_SLOT,
        };
        if let Some(slot) = self.free.pop() {
            self.nodes[slot as usize] = node;
            slot
        } else {
            self.nodes.push(node);
            (self.nodes.len() - 1) as u32
        }
    }
}

impl Default for LruLinkedList {
    fn default() -> Self {
        Self::new()
    }
}

impl ReplacementPolicy for LruLinkedList {
    fn name(&self) -> &'static str {
        "lru-linked-list"
    }

    fn on_hit(&mut self, page: PageRef) {
        let slot = self.slots.get(page);
        if slot != NO_SLOT {
            self.unlink(slot);
            self.link_back(slot);
        }
    }

    fn on_fault(&mut self, _page: PageRef) {}

    fn on_insert(&mut self, page: PageRef, _demand: bool) {
        let slot = self.slots.get(page);
        if slot != NO_SLOT {
            self.unlink(slot);
            self.link_back(slot);
            return;
        }
        let slot = self.alloc(page);
        self.link_back(slot);
        self.slots.set(page, slot);
    }

    fn take_victims(&mut self, need: usize, out: &mut Vec<PageRef>) {
        for _ in 0..need {
            let slot = self.head;
            if slot == NO_SLOT {
                break;
            }
            let page = self.nodes[slot as usize].page;
            self.unlink(slot);
            self.slots.clear(page);
            self.free.push(slot);
            out.push(page);
        }
    }

    fn on_region_closed(&mut self, region: RegionId) {
        for slot in self.slots.remove_region(region) {
            if slot != NO_SLOT {
                self.unlink(slot);
                self.free.push(slot);
            }
        }
    }
}

struct ResidentPage {
    page: PageRef,
    last_use: u64,
}

/// LRU that does no ordering work on a hit at all: it stamps the page with
/// a monotonically increasing counter and defers the ordering to eviction
/// time, where the coldest pages are picked out with a quickselect.
///
/// The stamp is a plain `u64` counter, deliberately *not* a clock reading —
/// LRU needs only the ordering of accesses, and `clock_gettime` (~20ns via
/// the vDSO, worse under virtualisation) would cost more than the linked
/// list this is meant to beat. A counter also never ties, so the resulting
/// order is unambiguous and identical to [`LruHashMap`]'s.
///
/// The trade is that `take_victims` is O(resident) rather than O(need).
/// That pays off here because the workload is overwhelmingly hit-dominated
/// (measured at ~2000 hits per eviction), and because the scan walks a
/// packed `Vec` sequentially — the hardware prefetcher handles it, unlike
/// the pointer-chasing a linked list does per hit. Note the scan is bounded
/// by the resident-page cap, not by region size: pages are kept in a dense
/// `Vec` and the [`SlotTable`] maps into it, so a large file with a small
/// cap costs no more than a small one.
pub struct LruCounter {
    resident: Vec<ResidentPage>,
    slots: SlotTable,
    clock: u64,
    /// Scratch buffers reused across `take_victims` calls.
    stamps: Vec<u64>,
    victims: Vec<(u64, u32)>,
}

impl LruCounter {
    #[must_use]
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            resident: Vec::with_capacity(capacity),
            slots: SlotTable::new(),
            clock: 0,
            stamps: Vec::with_capacity(capacity),
            victims: Vec::new(),
        }
    }

    /// Re-points the slot table at the current positions in `resident`,
    /// needed after anything that permutes or shifts that vector.
    fn reindex(&mut self) {
        for i in 0..self.resident.len() {
            let page = self.resident[i].page;
            self.slots.set(page, i as u32);
        }
    }
}

impl Default for LruCounter {
    fn default() -> Self {
        Self::new()
    }
}

impl ReplacementPolicy for LruCounter {
    fn name(&self) -> &'static str {
        "lru-counter"
    }

    fn on_hit(&mut self, page: PageRef) {
        let slot = self.slots.get(page);
        if slot != NO_SLOT {
            self.clock += 1;
            self.resident[slot as usize].last_use = self.clock;
        }
    }

    fn on_fault(&mut self, _page: PageRef) {}

    fn on_insert(&mut self, page: PageRef, _demand: bool) {
        self.clock += 1;
        let slot = self.slots.get(page);
        if slot != NO_SLOT {
            self.resident[slot as usize].last_use = self.clock;
            return;
        }
        self.slots.set(page, self.resident.len() as u32);
        self.resident.push(ResidentPage {
            page,
            last_use: self.clock,
        });
    }

    fn take_victims(&mut self, need: usize, out: &mut Vec<PageRef>) {
        let need = need.min(self.resident.len());
        if need == 0 {
            return;
        }
        // Find the cutoff stamp on a scratch copy of the stamps, never on
        // `resident` itself: permuting `resident` would invalidate every
        // survivor's slot and force an O(resident) `reindex` with scattered
        // `SlotTable` writes on each eviction. That reindex (plus a `drain`
        // shifting the whole vector) dominated at small page sizes, where
        // the resident count and the eviction rate are both high.
        let cutoff = if need < self.resident.len() {
            self.stamps.clear();
            self.stamps
                .extend(self.resident.iter().map(|entry| entry.last_use));
            *self.stamps.select_nth_unstable(need - 1).1
        } else {
            u64::MAX
        };
        // Stamps never tie, so exactly `need` entries satisfy this.
        self.victims.clear();
        self.victims.extend(
            self.resident
                .iter()
                .enumerate()
                .filter(|(_, entry)| entry.last_use <= cutoff)
                .map(|(i, entry)| (entry.last_use, i as u32)),
        );
        debug_assert_eq!(self.victims.len(), need);
        // True LRU order, exactly as `LruHashMap` emits it.
        self.victims.sort_unstable_by_key(|&(stamp, _)| stamp);
        for &(_, i) in &self.victims {
            let page = self.resident[i as usize].page;
            self.slots.clear(page);
            out.push(page);
        }
        // Remove highest index first: `swap_remove(i)` then always pulls in
        // a survivor (every victim above `i` is already gone), and only that
        // one moved entry needs its slot updated.
        self.victims
            .sort_unstable_by_key(|&(_, i)| std::cmp::Reverse(i));
        for &(_, i) in &self.victims {
            let i = i as usize;
            self.resident.swap_remove(i);
            if let Some(moved) = self.resident.get(i) {
                self.slots.set(moved.page, i as u32);
            }
        }
    }

    fn on_region_closed(&mut self, region: RegionId) {
        self.slots.remove_region(region);
        self.resident.retain(|entry| entry.page.region != region);
        self.reindex();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits(n: usize) -> Arc<[AtomicBool]> {
        (0..n).map(|_| AtomicBool::new(false)).collect()
    }

    #[test]
    fn clock_gives_referenced_pages_a_second_chance() {
        let mut clock = Clock::new();
        let refs = bits(4);
        clock.on_region_opened(R_A, Arc::clone(&refs));
        for i in 0..4 {
            clock.on_insert(p(R_A, i), false);
        }
        // Page 0 is the oldest but was read since insertion.
        refs[0].store(true, Ordering::Relaxed);
        let mut out = Vec::new();
        clock.take_victims(2, &mut out);
        assert_eq!(out, vec![p(R_A, 1), p(R_A, 2)]);
        // Its bit was consumed: next time it goes (after page 3).
        assert!(!refs[0].load(Ordering::Relaxed));
        out.clear();
        clock.take_victims(2, &mut out);
        assert_eq!(out, vec![p(R_A, 3), p(R_A, 0)]);
    }

    #[test]
    fn clock_terminates_when_every_page_is_referenced() {
        let mut clock = Clock::new();
        let refs = bits(3);
        clock.on_region_opened(R_A, Arc::clone(&refs));
        for i in 0..3 {
            clock.on_insert(p(R_A, i), true); // demand: bit set
        }
        let mut out = Vec::new();
        clock.take_victims(1, &mut out);
        // Full rotation clears all bits, then the original oldest goes.
        assert_eq!(out, vec![p(R_A, 0)]);
    }

    #[test]
    fn clock_purges_closed_regions() {
        let mut clock = Clock::new();
        clock.on_region_opened(R_A, bits(2));
        clock.on_region_opened(R_B, bits(2));
        clock.on_insert(p(R_A, 0), false);
        clock.on_insert(p(R_B, 0), false);
        clock.on_region_closed(R_A);
        let mut out = Vec::new();
        clock.take_victims(2, &mut out);
        assert_eq!(out, vec![p(R_B, 0)]);
    }

    const R_A: RegionId = 0x1000;
    const R_B: RegionId = 0x2000;

    fn p(region: RegionId, index: usize) -> PageRef {
        PageRef::new(region, index)
    }

    /// Feeds `pages` in as demand faults, then asks for `need` victims.
    fn insert_all(policy: &mut dyn ReplacementPolicy, pages: &[PageRef]) {
        for page in pages {
            policy.on_fault(*page);
            policy.on_insert(*page, true);
        }
    }

    fn take(policy: &mut dyn ReplacementPolicy, need: usize) -> Vec<PageRef> {
        let mut out = Vec::new();
        policy.take_victims(need, &mut out);
        out
    }

    #[test]
    fn fifo_evicts_oldest_first() {
        let mut policy = Fifo::new();
        insert_all(&mut policy, &[p(R_A, 0), p(R_A, 1), p(R_A, 2)]);
        assert_eq!(take(&mut policy, 2), vec![p(R_A, 0), p(R_A, 1)]);
        assert_eq!(take(&mut policy, 2), vec![p(R_A, 2)]);
    }

    /// The core ownership contract: a page handed out as a victim must not
    /// be handed out again until it is re-inserted. Violating this would let
    /// the pager decrement its resident count twice and underflow.
    #[test]
    fn victims_are_never_returned_twice() {
        for kind in [
            EvictionKind::Fifo,
            EvictionKind::Random,
            EvictionKind::PrefetchFirst,
            EvictionKind::TwoQueue,
            EvictionKind::LruHashMap,
        ] {
            let mut policy = build(kind, 42, 8);
            let pages: Vec<PageRef> = (0..8).map(|i| p(R_A, i)).collect();
            insert_all(policy.as_mut(), &pages);

            let mut seen = HashSet::new();
            let mut total = 0;
            // Ask for far more than exist: must drain to exactly 8, no
            // duplicates, and no infinite supply.
            for _ in 0..4 {
                for page in take(policy.as_mut(), 5) {
                    assert!(
                        seen.insert(page),
                        "{} returned {page:?} twice",
                        policy.name()
                    );
                    assert!(pages.contains(&page), "{} invented {page:?}", policy.name());
                    total += 1;
                }
            }
            assert_eq!(total, 8, "{} did not drain exactly once", policy.name());
        }
    }

    #[test]
    fn closing_a_region_purges_its_pages() {
        for kind in [
            EvictionKind::Fifo,
            EvictionKind::Random,
            EvictionKind::PrefetchFirst,
            EvictionKind::TwoQueue,
            EvictionKind::LruHashMap,
        ] {
            let mut policy = build(kind, 7, 8);
            insert_all(
                policy.as_mut(),
                &[p(R_A, 0), p(R_B, 0), p(R_A, 1), p(R_B, 1)],
            );
            policy.on_region_closed(R_A);

            let victims = take(policy.as_mut(), 8);
            assert_eq!(victims.len(), 2, "{} kept stale entries", policy.name());
            assert!(
                victims.iter().all(|page| page.region == R_B),
                "{} returned a page from the closed region: {victims:?}",
                policy.name()
            );
        }
    }

    #[test]
    fn prefetch_first_prefers_speculative_pages() {
        let mut policy = PrefetchFirst::new();
        // Two demand faults, each dragging in one speculative neighbour.
        policy.on_fault(p(R_A, 0));
        policy.on_insert(p(R_A, 0), true);
        policy.on_insert(p(R_A, 1), false);
        policy.on_fault(p(R_A, 4));
        policy.on_insert(p(R_A, 4), true);
        policy.on_insert(p(R_A, 5), false);

        // Both speculative pages go before either demand-faulted page.
        assert_eq!(take(&mut policy, 2), vec![p(R_A, 1), p(R_A, 5)]);
        assert_eq!(take(&mut policy, 2), vec![p(R_A, 0), p(R_A, 4)]);
    }

    /// A page evicted and then immediately re-faulted must be recognised as
    /// prematurely evicted and re-admitted as protected, which makes it
    /// outlive later probationary arrivals.
    #[test]
    fn twoq_promotes_on_ghost_hit() {
        let mut policy = TwoQueue::new(8);
        insert_all(&mut policy, &[p(R_A, 0), p(R_A, 1)]);

        // Evict page 0; it lands in the ghost list.
        assert_eq!(take(&mut policy, 1), vec![p(R_A, 0)]);
        assert_eq!(policy.ghost_hits(), 0);

        // It is needed again straight away: ghost hit, re-admitted protected.
        insert_all(&mut policy, &[p(R_A, 0)]);
        assert_eq!(policy.ghost_hits(), 1);

        // A fresh page arrives on probation.
        insert_all(&mut policy, &[p(R_A, 2)]);

        // Probation drains first: page 1 (older probationary) then page 2,
        // and only then the protected page 0.
        assert_eq!(
            take(&mut policy, 3),
            vec![p(R_A, 1), p(R_A, 2), p(R_A, 0)],
            "protected page was evicted before probationary ones"
        );
    }

    /// A page evicted long ago has fallen out of the bounded ghost list and
    /// must not be promoted, otherwise the protected queue grows without
    /// bound and the policy degenerates to FIFO-over-protected.
    #[test]
    fn twoq_ghost_list_is_bounded() {
        let mut policy = TwoQueue::new(2);
        insert_all(&mut policy, &[p(R_A, 0), p(R_A, 1), p(R_A, 2)]);
        // Evict all three; the ghost list only remembers the last two.
        assert_eq!(take(&mut policy, 3).len(), 3);

        insert_all(&mut policy, &[p(R_A, 0)]);
        assert_eq!(
            policy.ghost_hits(),
            0,
            "page 0 should have aged out of the 2-entry ghost list"
        );
    }

    #[test]
    fn twoq_hit_on_probationary_page_is_ignored() {
        let mut policy = TwoQueue::new(8);
        insert_all(&mut policy, &[p(R_A, 0), p(R_A, 1)]);
        policy.on_hit(p(R_A, 0)); // still in probation, not protected

        //Eviction order is unaffected: plain FIFO, as if the hit never happened
        assert_eq!(take(&mut policy, 2), vec![p(R_A, 0), p(R_A, 1)]);
    }

    #[test]
    fn twoq_clock_gives_referenced_page_a_second_chance() {
        let mut policy = TwoQueue::new(8);
        // Promote page 0, then page 1, into protected via ghost hits
        for page in [p(R_A, 0), p(R_A, 1)] {
            insert_all(&mut policy, &[page]);
            assert_eq!(take(&mut policy, 1), vec![page]); // evicted into ghost
            insert_all(&mut policy, &[page]); // ghost hit -> promoted
        }
        assert_eq!(policy.ghost_hits(), 2);

        policy.on_hit(p(R_A, 0)); // page 0 referenced; page 1 is not

        // probation is empty, so eviction must sweep protected
        // Page 0 gets a second chance and is requeued, page 1 is evicted first
        assert_eq!(take(&mut policy, 1), vec![p(R_A, 1)]);

        // Page 0's reference bit was consumed by that sweep, so it is evicted now
        assert_eq!(take(&mut policy, 1), vec![p(R_A, 0)]);
    }

    /// A hit on the least-recently-used page must move it out of harm's
    /// way; eviction order should then reflect access recency, not
    /// insertion order.
    #[test]
    fn lru_hit_moves_page_to_most_recently_used_end() {
        let mut policy = LruHashMap::new();
        insert_all(&mut policy, &[p(R_A, 0), p(R_A, 1), p(R_A, 2)]);

        // Touch page 0 again: it should no longer be the next victim.
        policy.on_hit(p(R_A, 0));

        assert_eq!(
            take(&mut policy, 3),
            vec![p(R_A, 1), p(R_A, 2), p(R_A, 0)],
            "hit page was evicted before untouched ones"
        );
    }

    #[test]
    fn random_is_reproducible_for_a_given_seed() {
        let run = |seed: u64| {
            let mut policy = Random::new(seed);
            insert_all(&mut policy, &(0..16).map(|i| p(R_A, i)).collect::<Vec<_>>());
            take(&mut policy, 8)
        };
        assert_eq!(run(1234), run(1234));
        assert_ne!(run(1234), run(9876));
    }

    #[test]
    fn eviction_kind_parses_and_round_trips() {
        for kind in [
            EvictionKind::Fifo,
            EvictionKind::Random,
            EvictionKind::PrefetchFirst,
            EvictionKind::TwoQueue,
            EvictionKind::LruHashMap,
            EvictionKind::LruLinkedList,
            EvictionKind::LruCounter,
            EvictionKind::Clock,
        ] {
            assert_eq!(EvictionKind::parse(kind.as_str()), Ok(kind));
            assert_eq!(build(kind, 0, 4).name(), kind.as_str());
        }
        assert_eq!(
            EvictionKind::parse("lru_hashmap"),
            Ok(EvictionKind::LruHashMap)
        );
        assert_eq!(
            EvictionKind::parse("lru_linked_list"),
            Ok(EvictionKind::LruLinkedList)
        );
        assert_eq!(
            EvictionKind::parse("lru_counter"),
            Ok(EvictionKind::LruCounter)
        );
        assert_eq!(
            EvictionKind::parse("second-chance"),
            Ok(EvictionKind::Clock)
        );
        assert!(EvictionKind::parse("arc").is_err());
    }

    /// The whole point of the two fast variants: they must be drop-in
    /// replacements for `LruHashMap`, differing only in cost. Drives a mixed
    /// insert/hit/evict workload across two regions and requires all three
    /// to name the same victims in the same order.
    #[test]
    fn all_lru_variants_agree_on_eviction_order() {
        fn drive(policy: &mut dyn ReplacementPolicy) -> Vec<PageRef> {
            let mut log = Vec::new();
            for i in 0..8 {
                policy.on_fault(p(R_A, i));
                policy.on_insert(p(R_A, i), true);
            }
            for i in 0..4 {
                policy.on_fault(p(R_B, i));
                policy.on_insert(p(R_B, i), i % 2 == 0);
            }
            // Touch a scattered subset, oldest-first, so recency order ends
            // up genuinely different from insertion order.
            for page in [p(R_A, 0), p(R_B, 1), p(R_A, 5), p(R_A, 0), p(R_B, 3)] {
                policy.on_hit(page);
            }
            log.extend(take(policy, 3));
            // Re-admit an evicted page and touch a survivor, then drain.
            policy.on_fault(p(R_A, 1));
            policy.on_insert(p(R_A, 1), true);
            policy.on_hit(p(R_A, 6));
            log.extend(take(policy, 20));
            log
        }

        let expected = drive(&mut LruHashMap::new());
        assert_eq!(expected.len(), 13, "workload should drain every page");
        assert_eq!(drive(&mut LruLinkedList::new()), expected);
        assert_eq!(drive(&mut LruCounter::new()), expected);
    }

    /// Closing a region must purge exactly its own pages and leave the
    /// other region's recency order intact.
    #[test]
    fn lru_variants_purge_only_the_closed_region() {
        fn drive(policy: &mut dyn ReplacementPolicy) -> Vec<PageRef> {
            insert_all(policy, &[p(R_A, 0), p(R_B, 0), p(R_A, 1), p(R_B, 1)]);
            policy.on_hit(p(R_B, 0));
            policy.on_region_closed(R_A);
            take(policy, 10)
        }

        let expected = vec![p(R_B, 1), p(R_B, 0)];
        assert_eq!(drive(&mut LruHashMap::new()), expected);
        assert_eq!(drive(&mut LruLinkedList::new()), expected);
        assert_eq!(drive(&mut LruCounter::new()), expected);
    }

    /// Slots freed by eviction are recycled, so a long run cannot grow the
    /// node arena or the resident vector without bound.
    #[test]
    fn lru_variants_recycle_slots_across_eviction_cycles() {
        for _ in 0..100 {
            let mut list = LruLinkedList::new();
            let mut counter = LruCounter::new();
            for round in 0..50usize {
                for policy in [
                    &mut list as &mut dyn ReplacementPolicy,
                    &mut counter as &mut dyn ReplacementPolicy,
                ] {
                    policy.on_fault(p(R_A, round));
                    policy.on_insert(p(R_A, round), true);
                    if round >= 4 {
                        assert_eq!(take(policy, 1).len(), 1);
                    }
                }
            }
            assert!(list.nodes.len() <= 5, "node arena grew unboundedly");
            assert!(counter.resident.len() <= 5, "resident vec grew unboundedly");
        }
    }
}

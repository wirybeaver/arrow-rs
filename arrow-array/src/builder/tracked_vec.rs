// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! A `Vec<T>` wrapper that keeps a [`MemoryPool`] reservation in sync with the
//! vector's capacity.

use arrow_buffer::{ArrowNativeType, Buffer, MutableBuffer};

#[cfg(feature = "pool")]
use arrow_buffer::{MemoryAllocationError, MemoryPool, MemoryReservation};

/// A `Vec<T>` that optionally tracks its capacity in a [`MemoryPool`].
///
/// When no pool is attached, all operations delegate directly to the inner
/// `Vec` with zero overhead.  When a pool is attached via [`attach_pool`] or
/// [`with_pool`], every capacity-changing operation updates the reservation so
/// the pool always reflects the builder's true memory footprint.
///
/// [`attach_pool`]: TrackedVec::attach_pool
/// [`with_pool`]: TrackedVec::with_pool
pub(crate) struct TrackedVec<T> {
    pub(crate) inner: Vec<T>,
    #[cfg(feature = "pool")]
    reservation: Option<Box<dyn MemoryReservation>>,
}

impl<T: std::fmt::Debug> std::fmt::Debug for TrackedVec<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.inner.fmt(f)
    }
}

impl<T> Default for TrackedVec<T> {
    fn default() -> Self {
        Self::with_capacity(0)
    }
}

impl<T> TrackedVec<T> {
    /// Creates a new `TrackedVec` backed by a `Vec` with the given capacity and
    /// no pool reservation.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: Vec::with_capacity(capacity),
            #[cfg(feature = "pool")]
            reservation: None,
        }
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    pub fn as_slice(&self) -> &[T] {
        self.inner.as_slice()
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.inner.as_mut_slice()
    }

    /// Appends `v`, tracking any capacity growth in the pool.
    #[inline]
    pub fn push(&mut self, v: T) {
        #[cfg(feature = "pool")]
        let old_cap = self.inner.capacity();
        self.inner.push(v);
        #[cfg(feature = "pool")]
        self.sync_reservation_if_grew(old_cap);
    }

    /// Extends from a slice, tracking any capacity growth in the pool.
    #[inline]
    pub fn extend_from_slice(&mut self, slice: &[T])
    where
        T: Copy,
    {
        #[cfg(feature = "pool")]
        let old_cap = self.inner.capacity();
        self.inner.extend_from_slice(slice);
        #[cfg(feature = "pool")]
        self.sync_reservation_if_grew(old_cap);
    }

    /// Reserves additional capacity, tracking any growth in the pool.
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        #[cfg(feature = "pool")]
        let old_cap = self.inner.capacity();
        self.inner.reserve(additional);
        #[cfg(feature = "pool")]
        self.sync_reservation_if_grew(old_cap);
    }

    /// Converts this `TrackedVec` into a [`MutableBuffer`], transferring the
    /// pool reservation into the buffer so tracking continues seamlessly.
    pub fn into_mutable_buffer(self) -> MutableBuffer
    where
        T: ArrowNativeType,
    {
        #[cfg(feature = "pool")]
        let reservation = self.reservation;
        #[cfg_attr(not(feature = "pool"), allow(unused_mut))]
        let mut buf = MutableBuffer::from(self.inner);
        #[cfg(feature = "pool")]
        buf.set_reservation(reservation);
        buf
    }

    /// Converts this `TrackedVec` into an immutable [`Buffer`], transferring the
    /// pool reservation into the buffer.
    pub fn into_buffer(self) -> Buffer
    where
        T: ArrowNativeType,
    {
        self.into_mutable_buffer().into()
    }

    #[cfg(feature = "pool")]
    #[inline]
    fn sync_reservation_if_grew(&mut self, old_cap: usize) {
        let new_cap = self.inner.capacity();
        if new_cap != old_cap {
            if let Some(ref mut res) = self.reservation {
                res.resize(new_cap * std::mem::size_of::<T>());
            }
        }
    }
}

#[cfg(feature = "pool")]
impl<T> TrackedVec<T> {
    /// Creates a new `TrackedVec` with the given capacity and registers it with
    /// the pool immediately.
    pub fn with_pool(capacity: usize, pool: &dyn MemoryPool) -> Result<Self, MemoryAllocationError> {
        let inner = Vec::with_capacity(capacity);
        let reservation = pool.try_reserve(inner.capacity() * std::mem::size_of::<T>())?;
        Ok(Self {
            inner,
            reservation: Some(reservation),
        })
    }

    /// Attaches a pool to an existing `TrackedVec`, reserving its current
    /// capacity.  Any prior reservation is replaced.
    pub fn attach_pool(&mut self, pool: &dyn MemoryPool) -> Result<(), MemoryAllocationError> {
        let bytes = self.inner.capacity() * std::mem::size_of::<T>();
        self.reservation = Some(pool.try_reserve(bytes)?);
        Ok(())
    }

    /// Fallible push: checks pool capacity before allowing Vec growth.
    #[inline]
    pub fn try_push(&mut self, v: T) -> Result<(), MemoryAllocationError> {
        if self.inner.len() == self.inner.capacity() {
            let new_cap = if self.inner.capacity() == 0 {
                4
            } else {
                self.inner.capacity() * 2
            };
            let new_bytes = new_cap * std::mem::size_of::<T>();
            if let Some(ref mut res) = self.reservation {
                res.try_resize(new_bytes)?;
            }
        }
        let old_cap = self.inner.capacity();
        self.inner.push(v);
        // Sync to actual capacity (Vec growth may differ from our prediction).
        self.sync_reservation_if_grew(old_cap);
        Ok(())
    }

    /// Fallible extend from slice: checks pool capacity before allowing Vec growth.
    #[inline]
    pub fn try_extend_from_slice(&mut self, slice: &[T]) -> Result<(), MemoryAllocationError>
    where
        T: Copy,
    {
        let needed = self.inner.len() + slice.len();
        if needed > self.inner.capacity() {
            let new_bytes = needed * std::mem::size_of::<T>();
            if let Some(ref mut res) = self.reservation {
                res.try_resize(new_bytes)?;
            }
        }
        let old_cap = self.inner.capacity();
        self.inner.extend_from_slice(slice);
        self.sync_reservation_if_grew(old_cap);
        Ok(())
    }

    /// Fallible reserve: checks pool capacity before reserving additional space.
    #[inline]
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), MemoryAllocationError> {
        let needed = self.inner.len() + additional;
        if needed > self.inner.capacity() {
            let new_bytes = needed * std::mem::size_of::<T>();
            if let Some(ref mut res) = self.reservation {
                res.try_resize(new_bytes)?;
            }
        }
        let old_cap = self.inner.capacity();
        self.inner.reserve(additional);
        self.sync_reservation_if_grew(old_cap);
        Ok(())
    }
}

impl<T> Extend<T> for TrackedVec<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        #[cfg(feature = "pool")]
        let old_cap = self.inner.capacity();
        self.inner.extend(iter);
        #[cfg(feature = "pool")]
        self.sync_reservation_if_grew(old_cap);
    }
}

impl<T> From<Vec<T>> for TrackedVec<T> {
    fn from(inner: Vec<T>) -> Self {
        Self {
            inner,
            #[cfg(feature = "pool")]
            reservation: None,
        }
    }
}

impl<T: ArrowNativeType> From<TrackedVec<T>> for Buffer {
    fn from(v: TrackedVec<T>) -> Self {
        v.into_buffer()
    }
}

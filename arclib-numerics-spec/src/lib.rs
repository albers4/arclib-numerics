// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

/// Specification: v1.0
pub trait ArrayLike {
    type Item;
    type Error;

    fn get(&self, index: usize) -> Result<&Self::Item, Self::Error>;
}

pub trait ArrayLikeMut: ArrayLike {
    fn set(&mut self, index: usize, value: Self::Item) -> Result<(), Self::Error>;
}


mod domain;
// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use ndarray::ArrayD;

pub mod bc;
pub mod checkpoint;
/// Specification: v0.1.0
pub mod domain;
pub mod kernel;
pub mod utils;

pub type Tensor = ArrayD<f32>;

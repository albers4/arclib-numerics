// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use crate::tensor::Tensor;

pub trait BcEvaluator: Send + Sync {
    fn apply(&self, state: &mut Tensor, mask: &Tensor, values: &Tensor);
}

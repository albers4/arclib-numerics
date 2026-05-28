// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use crate::Tensor;

pub trait BcEvaluator: Send + Sync {
    fn apply(&self, state: &mut Tensor, mask: &Tensor, values: &Tensor);
}

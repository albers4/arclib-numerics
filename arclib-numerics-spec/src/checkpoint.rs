// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use std::collections::HashMap;

use arclib_graph_spec::NodeId;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CheckpointStrategy {
    Stateless,
    Full,
    BoundaryOnly,
    RngSeed(u64),
    Revolve { interval: usize },
}

#[derive(Clone, Debug)]
pub struct CheckpointBuffer {
    pub strategy: CheckpointStrategy,
    pub data: Vec<u8>,
}

impl CheckpointBuffer {
    pub fn new(strategy: CheckpointStrategy, capacity: usize) -> Self {
        Self {
            strategy,
            data: Vec::with_capacity(capacity),
        }
    }

    pub fn empty(strategy: CheckpointStrategy) -> Self {
        Self {
            strategy,
            data: Vec::new(),
        }
    }
}

#[derive(Default)]
pub struct CheckpointManager {
    trajectory: Vec<HashMap<NodeId, CheckpointBuffer>>,
}

impl CheckpointManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn begin_step(&mut self) {
        self.trajectory.push(HashMap::new());
    }

    pub fn save(&mut self, node_id: NodeId, buffer: CheckpointBuffer) {
        if let Some(current_step) = self.trajectory.last_mut() {
            current_step.insert(node_id, buffer);
        } else {
            let mut map = HashMap::new();
            map.insert(node_id, buffer);
            self.trajectory.push(map);
        }
    }

    pub fn load(&mut self, node_id: &NodeId) -> Option<CheckpointBuffer> {
        self.trajectory.last_mut()?.remove(node_id)
    }

    pub fn end_step(&mut self) {
        self.trajectory.pop();
    }

    pub fn clear(&mut self) {
        self.trajectory.clear();
    }

    pub fn num_steps(&self) -> usize {
        self.trajectory.len()
    }
}

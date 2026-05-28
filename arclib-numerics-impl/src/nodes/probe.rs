// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use std::sync::Arc;

use arclib_graph_impl::fnv1a_hash;
use arclib_graph_spec::{GraphContext, Node, NodeId, Shape};
use arclib_numerics_spec::utils::ProbeExtractor;
use uuid::Uuid;

use crate::NumericsContextValue;

#[derive(Clone)]
pub struct ProbeNode {
    pub id: NodeId,
    pub source_id: NodeId,
    pub name: String,
    pub coordinates: Vec<Vec<usize>>,
    pub interval: usize,
    pub counter: usize,
    pub extractor: Arc<dyn ProbeExtractor>,
}

impl ProbeNode {
    pub fn new(
        name: &str,
        source_id: NodeId,
        coordinates: Vec<Vec<usize>>,
        interval: usize,
        extractor: Arc<dyn ProbeExtractor>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            source_id,
            name: name.to_string(),
            coordinates,
            interval,
            counter: 0,
            extractor,
        }
    }
}

impl Node<NumericsContextValue> for ProbeNode {
    fn type_id_static() -> u64
    where
        Self: Sized,
    {
        fnv1a_hash("ProbeNode")
    }

    fn id(&self) -> &NodeId {
        &self.id
    }

    fn compute(&mut self, ctx: &mut GraphContext<'_, NumericsContextValue>) {
        self.counter += 1;

        let state_arc = match ctx.temp.get(&self.source_id) {
            Some(NumericsContextValue::Tensor(t)) => t.clone(),
            _ => panic!("ProbeNode '{}': Source missing from temp", self.name),
        };

        if self.counter.is_multiple_of(self.interval) {
            println!("\n--- [Probe: {}] Step {} ---", self.name, self.counter);
            let output = self.extractor.extract(&state_arc, &self.coordinates);
            print!("{}", output);
        }

        ctx.temp
            .insert(self.id, NumericsContextValue::Tensor(state_arc));
    }

    fn dependencies(&self) -> Vec<NodeId> {
        vec![self.source_id]
    }

    fn infer_shape(&self, inputs: &[Shape]) -> Result<Shape, String> {
        if inputs.is_empty() {
            return Err("ProbeNode requires a source input".to_string());
        }
        Ok(inputs[0].clone())
    }

    fn as_node(&self) -> &dyn Node<NumericsContextValue> {
        self
    }

    fn as_node_mut(&mut self) -> &mut dyn Node<NumericsContextValue> {
        self
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn Node<NumericsContextValue>> {
        Box::new(self.clone())
    }
}

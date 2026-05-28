// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use std::sync::Arc;

use arclib_graph_impl::fnv1a_hash;
use arclib_graph_spec::{GraphContext, Node, NodeId, Shape};
use arclib_numerics_spec::utils::DataExporter;
use uuid::Uuid;

use crate::NumericsContextValue;

#[derive(Clone)]
pub struct ExportNode {
    pub id: NodeId,
    pub source_id: NodeId,
    pub interval: usize,
    pub counter: usize,
    pub base_path: String,
    pub exporter: Arc<dyn DataExporter>,
}

impl ExportNode {
    pub fn new(
        source_id: NodeId,
        interval: usize,
        base_path: &str,
        exporter: Arc<dyn DataExporter>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            source_id,
            interval,
            counter: 0,
            base_path: base_path.to_string(),
            exporter,
        }
    }
}

impl Node<NumericsContextValue> for ExportNode {
    fn type_id_static() -> u64
    where
        Self: Sized,
    {
        fnv1a_hash("ExportNode")
    }

    fn id(&self) -> &NodeId {
        &self.id
    }

    fn compute(&mut self, ctx: &mut GraphContext<'_, NumericsContextValue>) {
        self.counter += 1;

        let state_arc = match ctx.temp.get(&self.source_id) {
            Some(NumericsContextValue::Tensor(t)) => t.clone(),
            _ => panic!("ExportNode: Source missing from temp"),
        };

        if self.counter.is_multiple_of(self.interval) {
            println!("Exporting step {} to {}", self.counter, self.base_path);
            self.exporter
                .export(&state_arc, self.counter, &self.base_path);
        }

        ctx.temp
            .insert(self.id, NumericsContextValue::Tensor(state_arc));
    }

    fn dependencies(&self) -> Vec<NodeId> {
        vec![self.source_id]
    }

    fn infer_shape(&self, inputs: &[Shape]) -> Result<Shape, String> {
        if inputs.is_empty() {
            return Err("ExportNode requires a source input".to_string());
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

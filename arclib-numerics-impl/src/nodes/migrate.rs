// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use std::sync::Arc;

use arclib_graph_impl::fnv1a_hash;
use arclib_graph_spec::{GraphContext, Node, NodeId, Shape};
use arclib_numerics_spec::tensor::{Device, DeviceMemory, GpuBuffer, Tensor};
use ndarray::{ArrayD, IxDyn};
use std::ffi::c_void;
use uuid::Uuid;

use crate::NumericsContextValue;

unsafe extern "C" {
    unsafe fn cuda_allocate_vram(size_byte: usize) -> *mut c_void;
    unsafe fn cuda_upload(host_ptr: *const c_void, device_ptr: *mut c_void, size_bytes: usize);
    unsafe fn cuda_download(device_ptr: *const c_void, host_ptr: *mut c_void, size_bytes: usize);
}

#[derive(Clone)]
pub struct MigrateNode {
    pub id: NodeId,
    pub source_id: NodeId,
    pub target_device: Device,
}

impl MigrateNode {
    pub fn new(source_id: NodeId, target_device: Device) -> Self {
        Self {
            id: Uuid::new_v4(),
            source_id,
            target_device,
        }
    }
}

impl Node<NumericsContextValue> for MigrateNode {
    fn type_id_static() -> u64
    where
        Self: Sized,
    {
        fnv1a_hash("MigrateNode")
    }

    fn id(&self) -> &NodeId {
        &self.id
    }

    fn compute(&mut self, ctx: &mut GraphContext<'_, NumericsContextValue>) {
        let source_tensor = match ctx.temp.remove(&self.source_id) {
            Some(NumericsContextValue::Tensor(t)) => t,
            _ => panic!("MigrateNode: Source missing"),
        };

        if source_tensor.device == self.target_device {
            ctx.temp
                .insert(self.id, NumericsContextValue::Tensor(source_tensor));
            return;
        }

        // PCIe transfer
        let migrated_tensor = match (&source_tensor.memory.as_ref(), self.target_device) {
            // CPU -> GPU
            (DeviceMemory::Cpu(cpu_arr), Device::Gpu(gpu_id)) => {
                let size_bytes = cpu_arr.len() * std::mem::size_of::<f32>();

                let device_ptr = unsafe { cuda_allocate_vram(size_bytes) };
                if device_ptr.is_null() {
                    panic!(
                        "CUDA OOM: Failed to allocate {} bytes on GPU {}",
                        size_bytes, gpu_id
                    )
                }

                unsafe {
                    cuda_upload(cpu_arr.as_ptr() as *const c_void, device_ptr, size_bytes);
                }

                let gpu_buf = GpuBuffer {
                    handle: device_ptr,
                    size_bytes,
                };
                Tensor {
                    shape: source_tensor.shape.clone(),
                    device: Device::Gpu(gpu_id),
                    memory: Arc::new(DeviceMemory::Gpu(gpu_buf)),
                }
            }

            // GPU -> CPU
            (DeviceMemory::Gpu(gpu_buf), Device::Cpu) => {
                let mut cpu_arr = ArrayD::zeros(IxDyn(&source_tensor.shape.0));
                let size_bytes = gpu_buf.size_bytes;

                unsafe {
                    cuda_download(
                        gpu_buf.handle,
                        cpu_arr.as_mut_ptr() as *mut c_void,
                        size_bytes,
                    );
                }

                Tensor::from_cpu_array(cpu_arr)
            }

            _ => panic!("Unsupported migration path"),
        };

        ctx.temp.insert(
            self.id,
            NumericsContextValue::Tensor(Arc::new(migrated_tensor)),
        );
    }

    fn dependencies(&self) -> Vec<NodeId> {
        vec![self.source_id]
    }

    fn infer_shape(&self, inputs: &[Shape]) -> Result<Shape, String> {
        if inputs.is_empty() {
            return Err("MigrateNode requires a source".to_string());
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

// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use std::{ffi::c_void, sync::Arc};

use arclib_graph_spec::Shape;
use ndarray::{ArrayD, ArrayView, ArrayViewMut, IxDyn, SliceArg};

unsafe extern "C" {
    unsafe fn cuda_free_vram(device_ptr: *mut c_void);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
    Gpu(usize),
}

pub struct GpuBuffer {
    pub handle: *mut c_void,
    pub size_bytes: usize,
}

unsafe impl Send for GpuBuffer {}
unsafe impl Sync for GpuBuffer {}

impl Clone for GpuBuffer {
    fn clone(&self) -> Self {
        Self {
            handle: self.handle,
            size_bytes: self.size_bytes,
        }
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { cuda_free_vram(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

pub enum DeviceMemory {
    Cpu(ArrayD<f32>),
    Gpu(GpuBuffer),
}

impl Clone for DeviceMemory {
    fn clone(&self) -> Self {
        match self {
            DeviceMemory::Cpu(arr) => DeviceMemory::Cpu(arr.clone()),
            DeviceMemory::Gpu(buf) => DeviceMemory::Gpu(buf.clone()),
        }
    }
}

#[derive(Clone)]
pub struct Tensor {
    pub shape: Shape,
    pub device: Device,
    pub memory: Arc<DeviceMemory>,
}

impl Tensor {
    pub fn from_cpu_array(array: ArrayD<f32>) -> Self {
        let shape = Shape(array.shape().to_vec());
        Self {
            shape,
            device: Device::Cpu,
            memory: Arc::new(DeviceMemory::Cpu(array)),
        }
    }

    pub fn gpu_zeros(shape: Shape, device_id: usize) -> Self {
        let size_bytes = shape.0.iter().product::<usize>() * std::mem::size_of::<f32>();
        // TODO: Call cudaMalloc
        let buffer = GpuBuffer {
            handle: std::ptr::null_mut(),
            size_bytes,
        };

        Self {
            shape,
            device: Device::Gpu(device_id),
            memory: Arc::new(DeviceMemory::Gpu(buffer)),
        }
    }

    pub fn as_cpu(&self) -> &ArrayD<f32> {
        match self.memory.as_ref() {
            DeviceMemory::Cpu(arr) => arr,
            DeviceMemory::Gpu(_) => {
                panic!("Attempting to read GPU tensor as CPU array. Use MigrateNode first!")
            }
        }
    }

    pub fn make_cpu_mut(&mut self) -> &mut ArrayD<f32> {
        match Arc::make_mut(&mut self.memory) {
            DeviceMemory::Cpu(arr) => arr,
            DeviceMemory::Gpu(_) => {
                panic!("Attempting to mutate GPU tensor as CPU array. Use MigrateNode first!")
            }
        }
    }

    pub fn as_ptr(&self) -> *const f32 {
        self.as_cpu().as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        self.make_cpu_mut().as_mut_ptr()
    }

    pub fn slice<'a, S>(&'a self, info: S) -> ArrayView<'a, f32, S::OutDim>
    where
        S: SliceArg<IxDyn>,
    {
        self.as_cpu().slice(info)
    }

    pub fn slice_mut<'a, S>(&'a mut self, info: S) -> ArrayViewMut<'a, f32, S::OutDim>
    where
        S: SliceArg<IxDyn>,
    {
        self.make_cpu_mut().slice_mut(info)
    }
}

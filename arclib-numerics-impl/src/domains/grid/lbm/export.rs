// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use std::fs::{self, File};
use std::io::Write;
use std::marker::PhantomData;

use arclib_numerics_spec::tensor::Tensor;
use arclib_numerics_spec::utils::DataExporter;
use ndarray::s;

use crate::domains::grid::lbm::topology::LatticeTopology;

pub struct LbmVtkExporter<T: LatticeTopology> {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    _marker: PhantomData<T>,
}

impl<T: LatticeTopology> LbmVtkExporter<T> {
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        Self {
            nx,
            ny,
            nz,
            _marker: PhantomData,
        }
    }
}

impl<T: LatticeTopology> DataExporter for LbmVtkExporter<T> {
    fn export(&self, state: &Tensor, step: usize, base_path: &str) {
        fs::create_dir_all(base_path).unwrap();
        let filename = format!("{}/lbm_{:06}.vti", base_path, step);
        let mut file = File::create(filename).unwrap();

        writeln!(file, r#"<?xml version="1.0"?>"#).unwrap();
        writeln!(
            file,
            r#"<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">"#
        )
        .unwrap();
        if T::DIM == 2 {
            writeln!(
                file,
                r#"  <ImageData WholeExtent="0 {} 0 {} 0 0" Origin="0 0 0" Spacing="1 1 1">"#,
                self.nx - 1,
                self.ny - 1
            )
            .unwrap();
            writeln!(
                file,
                r#"      <Piece Extent="0 {} 0 {} 0 0">"#,
                self.nx - 1,
                self.ny - 1
            )
            .unwrap();
        } else {
            writeln!(
                file,
                r#"  <ImageData WholeExtent="0 {} 0 {} 0 {}" Origin="0 0 0" Spacing="1 1 1">"#,
                self.nx - 1,
                self.ny - 1,
                self.nz - 1,
            )
            .unwrap();
            writeln!(
                file,
                r#"      <Piece Extent="0 {} 0 {} 0 {}">"#,
                self.nx - 1,
                self.ny - 1,
                self.nz - 1
            )
            .unwrap();
        }

        writeln!(
            file,
            r#"          <PointData Scalars="Density" Vectors="Velocity">"#
        )
        .unwrap();

        // Write density
        writeln!(
            file,
            r#"              <DataArray type="Float32" Name="Density" format="ascii">"#
        )
        .unwrap();
        self.write_data(&mut file, state, false);
        writeln!(file, "\n            </DataArray>").unwrap();

        // Write velocity
        writeln!(file, r#"              <DataArray type="Float32" Name="Velocity" NumberOfComponents="3" format="ascii">"#).unwrap();
        self.write_data(&mut file, state, true);
        writeln!(file, "\n            </DataArray>").unwrap();

        writeln!(file, r#"          </PointData>"#).unwrap();
        writeln!(file, r#"      </Piece>"#).unwrap();
        writeln!(file, r#"  </ImageData>"#).unwrap();
        writeln!(file, r#"</VTKFile>"#).unwrap();
    }
}

impl<T: LatticeTopology> LbmVtkExporter<T> {
    fn write_data(&self, file: &mut File, state: &Tensor, is_velocity: bool) {
        let nz_iter = if T::DIM == 3 { self.nz } else { 1 };

        for z in 0..nz_iter {
            for y in 0..self.ny {
                for x in (0..self.nx).rev() {
                    let f_slice = match T::DIM {
                        2 => state.slice(s![x, y, ..]),
                        3 => state.slice(s![x, y, z, ..]),
                        _ => panic!("Unsupported DIM"),
                    };

                    let mut rho = 0.0;
                    let mut u = [0.0f32; 3];

                    for q in 0..T::Q {
                        let fq = f_slice[q];
                        rho += fq;
                        u[0] += fq * T::CX[q] as f32;
                        if T::DIM > 1 {
                            u[1] += fq * T::CY[q] as f32;
                        }
                        if T::DIM > 2 {
                            u[2] += fq * T::CZ[q] as f32;
                        }
                    }

                    if rho > 1e-6 {
                        u[0] /= rho;
                        u[1] /= rho;
                        u[2] /= rho;
                    }

                    if is_velocity {
                        write!(file, "{} {} {} ", u[0], u[1], u[2]).unwrap();
                    } else {
                        write!(file, "{} ", rho).unwrap();
                    }
                }
            }
        }
    }
}

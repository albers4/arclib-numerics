// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use std::fs::{self, File};
use std::io::Write;

use arclib_numerics_spec::{Tensor, utils::DataExporter};
use ndarray::s;

pub struct LbmVtkExporter {
    pub nx: usize,
    pub ny: usize,
}

impl LbmVtkExporter {
    pub fn new(nx: usize, ny: usize) -> Self {
        Self { nx, ny }
    }
}

impl DataExporter for LbmVtkExporter {
    fn export(&self, state: &Tensor, step: usize, base_path: &str) {
        fs::create_dir_all(base_path).unwrap();
        let filename = format!("{}/lbm_{:06}.vti", base_path, step);
        let mut file = File::create(filename).unwrap();

        // D2Q9 velocities
        let cx = [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0];
        let cy = [0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0];

        writeln!(file, r#"<?xml version="1.0"?>"#).unwrap();
        writeln!(
            file,
            r#"<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">"#
        )
        .unwrap();
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
        for y in 0..self.ny {
            for x in 0..self.nx {
                let f = state.slice(s![x, y, ..]);
                let rho: f32 = f.iter().sum();
                write!(file, "{} ", rho).unwrap();
            }
        }
        writeln!(file, "\n            </DataArray>").unwrap();

        // Write velocity
        writeln!(file, r#"              <DataArray type="Float32" Name="Velocity" NumberOfComponents="3" format="ascii">"#).unwrap();
        for y in 0..self.ny {
            for x in 0..self.nx {
                let f = state.slice(s![x, y, ..]);
                let mut rho = 0.0;
                let mut mux = 0.0;
                let mut muy = 0.0;
                for q in 0..9 {
                    rho += f[q];
                    mux += f[q] * cx[q];
                    muy += f[q] * cy[q];
                }
                let ux = if rho > 1e-6 { mux / rho } else { 0.0 };
                let uy = if rho > 1e-6 { muy / rho } else { 0.0 };

                write!(file, "{} {} 0.0 ", ux, uy).unwrap();
            }
        }
        writeln!(file, "\n            </DataArray>").unwrap();

        writeln!(file, r#"          </PointData>"#).unwrap();
        writeln!(file, r#"      </Piece>"#).unwrap();
        writeln!(file, r#"  </ImageData>"#).unwrap();
        writeln!(file, r#"</VTKFile>"#).unwrap();
    }
}

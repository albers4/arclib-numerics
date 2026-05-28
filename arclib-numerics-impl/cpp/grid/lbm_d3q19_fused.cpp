// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

#include <stddef.h>
#include <stdint.h>
#include <omp.h>

extern "C" {

static const int cx[19] = {0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0};
static const int cy[19] = {0, 0, 0, 1,-1, 0, 0, 1, 1,-1,-1, 0, 0, 0, 0, 1,-1, 1,-1};
static const int cz[19] = {0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1, 1,-1,-1, 1, 1,-1,-1};
static const int opp[19] = {0, 2, 1, 4, 3, 6, 5, 10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15};

static const float w[19] = {
    1.0f/3.0f,
    1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f
};

void lbm_d3q19_fused_forward(
    const float* f_in,
    float* f_out,
    const float* solid,
    float omega,
    int nx, int ny, int nz
) {
    size_t total_size = (size_t)nx * ny * nz * 19;
    for (size_t i = 0; i < total_size; i++) {
        f_out[i] = f_in[i];
    }

    #pragma omp parallel for collapse(3) schedule(static)
    for (int x = 0; x < nx; x++) {
        for (int y = 0; y < ny; y++) {
            for (int z = 0; z < nz; z++) {
                
                int idx = (x * ny + y) * nz + z;

                if (solid[idx] > 0.5f) {
                    continue;
                }

                float rho = 0.0f;
                float ux = 0.0f, uy = 0.0f, uz = 0.0f;

                for (int q = 0; q < 19; q++) {
                    float f_q = f_in[idx * 19 + q];
                    rho += f_q;
                    ux += f_q * cx[q];
                    uy += f_q * cy[q];
                    uz += f_q * cz[q];
                }
                
                if (rho < 1e-6f) {
                    rho = 1.0f; ux = 0.0f; uy = 0.0f; uz = 0.0f;
                } else {
                    ux /= rho; uy /= rho; uz /= rho;
                }
                
                float usqr = ux * ux + uy * uy + uz * uz;

                for (int q = 0; q < 19; q++) {
                    float cu = cx[q] * ux + cy[q] * uy + cz[q] * uz;
                    float feq = w[q] * rho * (1.0f + 3.0f*cu + 4.5f*cu*cu - 1.5f*usqr);
                    
                    float f_post_coll = f_in[idx * 19 + q] + omega * (feq - f_in[idx * 19 + q]);
                    
                    int x_dst = x + cx[q];
                    int y_dst = y + cy[q];
                    int z_dst = z + cz[q];
                    
                    if (x_dst < 0 || x_dst >= nx || y_dst < 0 || y_dst >= ny || z_dst < 0 || z_dst >= nz) {
                        continue; 
                    }
                    
                    int idx_dst = (x_dst * ny + y_dst) * nz + z_dst;
                    
                    if (solid[idx_dst] < 0.5f) {
                        f_out[idx_dst * 19 + q] = f_post_coll;
                    } else {
                        int opp_q = opp[q];
                        f_out[idx * 19 + opp_q] = f_post_coll;
                    }
                }
            }
        }
    }
}

void lbm_d3q19_fused_backward(
    const float* f_in,
    const float* grad_f_out,
    float* grad_f_in,
    const float* solid,
    float omega,
    int nx, int ny, int nz
) {
    #pragma omp parallel for collapse(3) schedule(static)
    for (int x = 0; x < nx; x++) {
        for (int y = 0; y < ny; y++) {
            for (int z = 0; z < nz; z++) {
                int idx = (x * ny + y) * nz + z;
                
                if (solid[idx] > 0.5f) {
                    for (int q = 0; q < 19; q++) {
                        grad_f_in[idx * 19 + q] = grad_f_out[idx * 19 + q];
                    }
                    continue;
                }
                
                float grad_post[19];
                for (int q = 0; q < 19; q++) {
                    int x_dst = x + cx[q];
                    int y_dst = y + cy[q];
                    int z_dst = z + cz[q];
                    
                    if (x_dst < 0 || x_dst >= nx || y_dst < 0 || y_dst >= ny || z_dst < 0 || z_dst >= nz) {
                        grad_post[q] = 0.0f; 
                    } else {
                        int idx_dst = (x_dst * ny + y_dst) * nz + z_dst;
                        if (solid[idx_dst] < 0.5f) {
                            grad_post[q] = grad_f_out[idx_dst * 19 + q];
                        } else {
                            int opp_q = opp[q];
                            grad_post[q] = grad_f_out[idx * 19 + opp_q];
                        }
                    }
                }
                
                float rho = 0.0f, ux = 0.0f, uy = 0.0f, uz = 0.0f;
                for (int k = 0; k < 19; k++) {
                    float f_k = f_in[idx * 19 + k];
                    rho += f_k;
                    ux += f_k * cx[k];
                    uy += f_k * cy[k];
                    uz += f_k * cz[k];
                }
                if (rho < 1e-6f) { rho = 1.0f; ux = 0.0f; uy = 0.0f; uz = 0.0f; }
                else { ux /= rho; uy /= rho; uz /= rho; }
                float usqr = ux * ux + uy * uy + uz * uz;
                
                float E[19], Hx[19], Hy[19], Hz[19];
                for (int p = 0; p < 19; p++) {
                    float cu = cx[p] * ux + cy[p] * uy + cz[p] * uz;
                    E[p] = 1.0f + 3.0f * cu + 4.5f * cu * cu - 1.5f * usqr;
                    Hx[p] = 3.0f * cx[p] + 9.0f * cu * cx[p] - 3.0f * ux;
                    Hy[p] = 3.0f * cy[p] + 9.0f * cu * cy[p] - 3.0f * uy;
                    Hz[p] = 3.0f * cz[p] + 9.0f * cu * cz[p] - 3.0f * uz;
                }
                
                for (int q = 0; q < 19; q++) {
                    float sum_p = 0.0f;
                    for (int p = 0; p < 19; p++) {
                        float df_eq_p_df_q = w[p] * (E[p] + Hx[p] * (cx[q] - ux) + Hy[p] * (cy[q] - uy) + Hz[p] * (cz[q] - uz));
                        sum_p += grad_post[p] * df_eq_p_df_q;
                    }
                    grad_f_in[idx * 19 + q] = (1.0f - omega) * grad_post[q] + omega * sum_p;
                }
            }
        }
    }
}

} // extern "C"
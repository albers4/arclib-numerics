#include <stddef.h>
#include <stdint.h>
#include <omp.h>

extern "C" {

// D2Q9 velocity directions and weights
static const int cx[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
static const int cy[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
static const int opp[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};
static const float w[9] = {4.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 
                           1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f};

void lbm_d2q9_fused_forward(
    const float* f_in,
    float* f_out,
    const float* solid,
    float omega,
    int nx, int ny
) {
    for (int i = 0; i < nx * ny * 9; i++) {
        f_out[i] = f_in[i];
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int x = 0; x < nx; x++) {
        for (int y = 0; y < ny; y++) {
            int idx = x * ny + y;

            if (solid[idx] > 0.5f) {
                continue;
            }

            // Compute macroscopic moments at the CURRENT node
            float rho = 0.0f;
            float ux = 0.0f;
            float uy = 0.0f;

            for (int q = 0; q < 9; q++) {
                float f_q = f_in[idx * 9 + q];
                rho += f_q;
                ux += f_q * cx[q];
                uy += f_q * cy[q];
            }
            
            // Safety check to prevent division by zero
            if (rho < 1e-6f) {
                rho = 1.0f; ux = 0.0f; uy = 0.0f;
            } else {
                ux /= rho;
                uy /= rho;
            }
            
            float usqr = ux * ux + uy * uy;

            for (int q = 0; q < 9; q++) {
                float cu = cx[q] * ux + cy[q] * uy;
                float feq = w[q] * rho * (1.0f + 3.0f*cu + 4.5f*cu*cu - 1.5f*usqr);
                
                float f_post_coll = f_in[idx * 9 + q] + omega * (feq - f_in[idx * 9 + q]);
                
                int x_dst = x + cx[q];
                int y_dst = y + cy[q];
                
                if (x_dst < 0 || x_dst >= nx || y_dst < 0 || y_dst >= ny) {
                    continue; 
                }
                
                int idx_dst = x_dst * ny + y_dst;
                
                if (solid[idx_dst] < 0.5f) {
                    f_out[idx_dst * 9 + q] = f_post_coll;
                } else {
                    int opp_q = opp[q];
                    f_out[idx * 9 + opp_q] = f_post_coll;
                }
            }
        }
    }
}

void lbm_d2q9_fused_backward(
    const float* f_in,
    const float* grad_f_out,
    float* grad_f_in,
    const float* solid,
    float omega,
    int nx, int ny
) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int x = 0; x < nx; x++) {
        for (int y = 0; y < ny; y++) {
            int idx = x * ny + y;
            
            // Solid nodes just pass the gradient through (identity Jacobian)
            if (solid[idx] > 0.5f) {
                for (int q = 0; q < 9; q++) {
                    grad_f_in[idx * 9 + q] = grad_f_out[idx * 9 + q];
                }
                continue;
            }
            
            float grad_post[9];
            for (int q = 0; q < 9; q++) {
                int x_dst = x + cx[q];
                int y_dst = y + cy[q];
                
                if (x_dst < 0 || x_dst >= nx || y_dst < 0 || y_dst >= ny) {
                    grad_post[q] = 0.0f; // Dropped at boundary in forward pass
                } else {
                    int idx_dst = x_dst * ny + y_dst;
                    if (solid[idx_dst] < 0.5f) {
                        grad_post[q] = grad_f_out[idx_dst * 9 + q];
                    } else {
                        // Bounce-back adjoint
                        int opp_q = opp[q];
                        grad_post[q] = grad_f_out[idx * 9 + opp_q];
                    }
                }
            }
            
            float rho = 0.0f, ux = 0.0f, uy = 0.0f;
            for (int k = 0; k < 9; k++) {
                float f_k = f_in[idx * 9 + k];
                rho += f_k;
                ux += f_k * cx[k];
                uy += f_k * cy[k];
            }
            if (rho < 1e-6f) { rho = 1.0f; ux = 0.0f; uy = 0.0f; }
            else { ux /= rho; uy /= rho; }
            float usqr = ux * ux + uy * uy;
            
            // Precompute E_p and the derivative of E_p w.r.t velocity (H_p)
            float E[9], Hx[9], Hy[9];
            for (int p = 0; p < 9; p++) {
                float cu = cx[p] * ux + cy[p] * uy;
                E[p] = 1.0f + 3.0f * cu + 4.5f * cu * cu - 1.5f * usqr;
                Hx[p] = 3.0f * cx[p] + 9.0f * cu * cx[p] - 3.0f * ux;
                Hy[p] = 3.0f * cy[p] + 9.0f * cu * cy[p] - 3.0f * uy;
            }
            
            // Apply the chain rule: grad_f_in = (1-w)*grad_post + w * (grad_post^T * df_eq/df_in)
            for (int q = 0; q < 9; q++) {
                float sum_p = 0.0f;
                for (int p = 0; p < 9; p++) {
                    float df_eq_p_df_q = w[p] * (E[p] + Hx[p] * (cx[q] - ux) + Hy[p] * (cy[q] - uy));
                    sum_p += grad_post[p] * df_eq_p_df_q;
                }
                grad_f_in[idx * 9 + q] = (1.0f - omega) * grad_post[q] + omega * sum_p;
            }
        }
    }
}

} // extern "C"
use rayon::prelude::*;

use crate::marihydro::infra::constants::tolerances;
use crate::marihydro::infra::error::{MhError, MhResult};

#[derive(Clone)]
struct Workspace {
    a: Vec<f64>,
    b: Vec<f64>,
    c: Vec<f64>,
    d: Vec<f64>,
}

impl Workspace {
    fn new(size: usize) -> Self {
        Self {
            a: vec![0.0; size],
            b: vec![0.0; size],
            c: vec![0.0; size],
            d: vec![0.0; size],
        }
    }
}

pub fn apply_diffusion_adi(
    u_in: &[f64],
    u_out: &mut [f64],
    u_star: &mut [f64],
    nx: usize,
    ny: usize,
    ng: usize,
    dx: f64,
    dy: f64,
    nu: f64,
    dt: f64,
) -> MhResult<()> {
    validate_diffusion_params(nu, dt, dx, dy)?;

    let rx = nu * dt * 0.5 / (dx * dx);
    let ry = nu * dt * 0.5 / (dy * dy);
    let stride = nx + 2 * ng;

    log::debug!("扩散步进: rx={:.4}, ry={:.4}", rx, ry);

    let expected_len = stride * (ny + 2 * ng);
    if u_in.len() != expected_len || u_out.len() != expected_len || u_star.len() != expected_len {
        return Err(MhError::InvalidInput(format!(
            "数组尺寸不匹配: 期望 {}, 实际 in={}, out={}, star={}",
            expected_len,
            u_in.len(),
            u_out.len(),
            u_star.len()
        )));
    }

    let mut workspaces_x: Vec<Workspace> = (0..ny).map(|_| Workspace::new(nx)).collect();

    (ng..ny + ng)
        .into_par_iter()
        .zip(workspaces_x.par_iter_mut())
        .try_for_each(|(j, ws)| -> MhResult<()> {
            let offset = j * stride;

            for i in 0..nx {
                let idx = offset + (i + ng);

                let u_curr = u_in[idx];
                let diffusion_y = ry * (u_in[idx + stride] - 2.0 * u_curr + u_in[idx - stride]);

                ws.a[i] = -rx;
                ws.b[i] = 1.0 + 2.0 * rx;
                ws.c[i] = -rx;
                ws.d[i] = u_curr + diffusion_y;
            }

            apply_neumann_bc_x(&mut ws.a, &mut ws.b, &mut ws.c, nx);

            thomas_solve_inplace(&ws.a, &ws.b, &ws.c, &mut ws.d)?;

            for i in 0..nx {
                u_star[offset + i + ng] = ws.d[i];
            }

            Ok(())
        })?;

    let mut workspaces_y: Vec<Workspace> = (0..nx).map(|_| Workspace::new(ny)).collect();

    (ng..nx + ng)
        .into_par_iter()
        .zip(workspaces_y.par_iter_mut())
        .try_for_each(|(i, ws)| -> MhResult<()> {
            for j in 0..ny {
                let idx = (j + ng) * stride + i;

                let u_curr = u_star[idx];
                let diffusion_x = rx * (u_star[idx + 1] - 2.0 * u_curr + u_star[idx - 1]);

                ws.a[j] = -ry;
                ws.b[j] = 1.0 + 2.0 * ry;
                ws.c[j] = -ry;
                ws.d[j] = u_curr + diffusion_x;
            }

            apply_neumann_bc_y(&mut ws.a, &mut ws.b, &mut ws.c, ny);

            thomas_solve_inplace(&ws.a, &ws.b, &ws.c, &mut ws.d)?;

            for j in 0..ny {
                u_out[(j + ng) * stride + i] = ws.d[j];
            }

            Ok(())
        })?;

    Ok(())
}

#[inline]
fn apply_neumann_bc_x(a: &mut [f64], b: &mut [f64], c: &mut [f64], nx: usize) {
    b[0] += a[0];
    a[0] = 0.0;

    b[nx - 1] += c[nx - 1];
    c[nx - 1] = 0.0;
}

#[inline]
fn apply_neumann_bc_y(a: &mut [f64], b: &mut [f64], c: &mut [f64], ny: usize) {
    b[0] += a[0];
    a[0] = 0.0;

    b[ny - 1] += c[ny - 1];
    c[ny - 1] = 0.0;
}

fn validate_diffusion_params(nu: f64, dt: f64, dx: f64, dy: f64) -> MhResult<()> {
    if nu < 0.0 || nu > 100.0 {
        return Err(MhError::InvalidInput(format!(
            "扩散系数异常: ν = {:.3} m²/s",
            nu
        )));
    }

    if dt <= 0.0 || dt > 3600.0 {
        return Err(MhError::InvalidInput(format!(
            "时间步长异常: dt = {:.3} s",
            dt
        )));
    }

    if dx <= 0.0 || dy <= 0.0 {
        return Err(MhError::InvalidInput(format!(
            "网格间距异常: dx={}, dy={}",
            dx, dy
        )));
    }

    let rx = nu * dt / (dx * dx);
    let ry = nu * dt / (dy * dy);
    if rx > 10.0 || ry > 10.0 {
        log::warn!(
            "扩散 CFL 数过大 (rx={:.2}, ry={:.2})，可能导致精度损失",
            rx,
            ry
        );
    }

    Ok(())
}

fn thomas_solve_inplace(a: &[f64], b: &[f64], c: &[f64], d: &mut [f64]) -> MhResult<()> {
    let n = d.len();
    let mut c_prime = vec![0.0; n];

    let denom = b[0];
    if denom.abs() < tolerances::EPSILON {
        return Err(MhError::Runtime(format!(
            "Thomas 算法失败: 矩阵奇异 (b[0]={:.2e})",
            denom
        )));
    }

    c_prime[0] = c[0] / denom;
    d[0] = d[0] / denom;

    for i in 1..n {
        let denom = b[i] - a[i] * c_prime[i - 1];
        if denom.abs() < tolerances::EPSILON {
            return Err(MhError::Runtime(format!(
                "Thomas 算法失败: 矩阵奇异 (索引 {})",
                i
            )));
        }

        if i < n - 1 {
            c_prime[i] = c[i] / denom;
        }
        d[i] = (d[i] - a[i] * d[i - 1]) / denom;
    }

    for i in (0..n - 1).rev() {
        d[i] -= c_prime[i] * d[i + 1];
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diffusion_smoothing() {
        let nx = 10;
        let ny = 10;
        let ng = 2;
        let stride = nx + 2 * ng;

        let mut u_in = vec![0.0; stride * (ny + 2 * ng)];
        let mut u_out = vec![0.0; u_in.len()];
        let mut u_star = vec![0.0; u_in.len()];

        let center_idx = (ny / 2 + ng) * stride + (nx / 2 + ng);
        u_in[center_idx] = 100.0;

        apply_diffusion_adi(
            &u_in,
            &mut u_out,
            &mut u_star,
            nx,
            ny,
            ng,
            1.0,
            1.0,
            1.0,
            1.0,
        )
        .unwrap();

        let peak = u_out[center_idx];
        let neighbor = u_out[center_idx + 1];

        assert!(peak < 100.0);
        assert!(neighbor > 0.0);
    }

    #[test]
    fn test_thomas_solver() {
        let a = vec![0.0, -1.0, -1.0];
        let b = vec![2.0, 2.0, 2.0];
        let c = vec![-1.0, -1.0, 0.0];
        let mut d = vec![1.0, 0.0, 1.0];

        thomas_solve_inplace(&a, &b, &c, &mut d).unwrap();

        assert!((d[0] - 0.75).abs() < 1e-10);
        assert!((d[1] - 0.5).abs() < 1e-10);
        assert!((d[2] - 0.75).abs() < 1e-10);
    }
}

//! 梯度计算算子

use ndarray::Array2;
use rayon::prelude::*;

use crate::marihydro::domain::mesh::Mesh;
use crate::marihydro::infra::error::MhResult;

pub struct ScalarGradient {
    pub grad_x: Array2<f64>,
    pub grad_y: Array2<f64>,
}

pub fn compute_gradient(field: &Array2<f64>, mesh: &Mesh) -> MhResult<ScalarGradient> {
    let (dx, dy) = mesh.transform.resolution();
    let (total_ny, total_nx) = mesh.total_size();

    if field.dim() != (total_ny, total_nx) {
        return Err(crate::marihydro::infra::error::MhError::InvalidInput(
            format!(
                "场尺寸{:?}与网格{:?}不匹配",
                field.dim(),
                (total_ny, total_nx)
            ),
        ));
    }

    let mut grad_x = Array2::zeros((total_ny, total_nx));
    let mut grad_y = Array2::zeros((total_ny, total_nx));

    let inv_2dx = 0.5 / dx;
    let inv_2dy = 0.5 / dy;

    let stride = total_nx;
    let field_slice = field.as_slice_memory_order().ok_or_else(|| {
        crate::marihydro::infra::error::MhError::InternalError("场数组内存非连续".into())
    })?;

    let grad_x_slice = grad_x.as_slice_memory_order_mut().ok_or_else(|| {
        crate::marihydro::infra::error::MhError::InternalError("grad_x内存非连续".into())
    })?;

    let grad_y_slice = grad_y.as_slice_memory_order_mut().ok_or_else(|| {
        crate::marihydro::infra::error::MhError::InternalError("grad_y内存非连续".into())
    })?;

    mesh.active_indices.par_iter().for_each(|(j, i)| {
        let idx = j * stride + i;

        unsafe {
            let phi_e = *field_slice.get_unchecked(idx + 1);
            let phi_w = *field_slice.get_unchecked(idx - 1);
            let phi_n = *field_slice.get_unchecked(idx + stride);
            let phi_s = *field_slice.get_unchecked(idx - stride);

            *grad_x_slice.get_unchecked_mut(idx) = (phi_e - phi_w) * inv_2dx;
            *grad_y_slice.get_unchecked_mut(idx) = (phi_n - phi_s) * inv_2dy;
        }
    });

    Ok(ScalarGradient { grad_x, grad_y })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_placeholder() {
        // 完整测试需要mock mesh
        assert!(true);
    }
}

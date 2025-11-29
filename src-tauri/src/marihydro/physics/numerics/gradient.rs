use crate::marihydro::infra::error::MhResult;

pub struct ScalarGradient {
    pub grad_x: Vec<f64>,
    pub grad_y: Vec<f64>,
}

impl ScalarGradient {
    pub fn new(n_cells: usize) -> Self {
        Self {
            grad_x: vec![0.0; n_cells],
            grad_y: vec![0.0; n_cells],
        }
    }

    pub fn reset(&mut self) {
        self.grad_x.iter_mut().for_each(|x| *x = 0.0);
        self.grad_y.iter_mut().for_each(|y| *y = 0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_creation() {
        let grad = ScalarGradient::new(100);
        assert_eq!(grad.grad_x.len(), 100);
        assert_eq!(grad.grad_y.len(), 100);
    }

    #[test]
    fn test_gradient_reset() {
        let mut grad = ScalarGradient::new(10);
        grad.grad_x[0] = 1.0;
        grad.grad_y[0] = 2.0;
        grad.reset();
        assert_eq!(grad.grad_x[0], 0.0);
        assert_eq!(grad.grad_y[0], 0.0);
    }
}

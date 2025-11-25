// src-tauri/src/marihydro/io/loaders/raster.rs

use super::RasterLoader;
use crate::marihydro::infra::error::{MhError, MhResult};
use ndarray::Array2;
use std::path::Path;

/// 标准栅格加载器
pub struct StandardRasterLoader;

impl RasterLoader for StandardRasterLoader {
    fn load_array(
        &self,
        file_path: &str,
        target_shape: (usize, usize),
        _target_bounds: Option<(f64, f64, f64, f64)>,
    ) -> MhResult<Array2<f64>> {
        let path = Path::new(file_path);

        if !path.exists() {
            return Err(MhError::Io {
                context: format!("文件不存在: {}", file_path),
                source: std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("文件不存在: {}", file_path),
                ),
            });
        }

        // TODO: [工程化接入点] 集成 gdal crate
        // 目前阶段为了不破坏编译环境 (GDAL 需要系统库支持)，
        // 我们暂时返回一个 Mock 的数据，或者简单的 ASC 读取器。

        // 真正的逻辑应该是：
        // 1. let dataset = gdal::Dataset::open(path)?;
        // 2. dataset.rasterband(1)?.read_as_array(...)
        // 3. 执行重采样算法 (Bilinear/Cubic) 匹配 target_shape

        println!("(WARN) 使用模拟加载器读取: {}", file_path);

        // 模拟返回一个平坦地形 (用于测试)
        let (nx, ny) = target_shape;
        let array = Array2::from_elem((ny, nx), -10.0); // 水深 10m

        Ok(array)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockRasterLoader {
        shape: (usize, usize),
        fill_value: f64,
        value_generator: Option<Box<dyn Fn(usize, usize, usize, usize) -> f64>>,
        should_fail: bool,
    }

    struct MockRasterLoaderBuilder {
        shape: (usize, usize),
        fill_value: f64,
        value_generator: Option<Box<dyn Fn(usize, usize, usize, usize) -> f64>>,
        should_fail: bool,
    }

    impl MockRasterLoader {
        fn new() -> MockRasterLoaderBuilder {
            MockRasterLoaderBuilder {
                shape: (10, 10),
                fill_value: 0.0,
                value_generator: None,
                should_fail: false,
            }
        }
    }

    impl MockRasterLoaderBuilder {
        fn shape(mut self, shape: (usize, usize)) -> Self {
            self.shape = shape;
            self
        }

        fn fill_value(mut self, value: f64) -> Self {
            self.fill_value = value;
            self
        }

        fn with_generator<F>(mut self, func: F) -> Self
        where
            F: Fn(usize, usize, usize, usize) -> f64 + 'static,
        {
            self.value_generator = Some(Box::new(func));
            self
        }

        fn should_fail(mut self, fail: bool) -> Self {
            self.should_fail = fail;
            self
        }

        fn build(self) -> MockRasterLoader {
            MockRasterLoader {
                shape: self.shape,
                fill_value: self.fill_value,
                value_generator: self.value_generator,
                should_fail: self.should_fail,
            }
        }
    }

    impl RasterLoader for MockRasterLoader {
        fn load_array(
            &self,
            _file_path: &str,
            _target_shape: (usize, usize),
            _target_bounds: Option<(f64, f64, f64, f64)>,
        ) -> MhResult<Array2<f64>> {
            if self.should_fail {
                return Err(MhError::Io {
                    context: "Mock I/O error".to_string(),
                    source: std::io::Error::new(std::io::ErrorKind::Other, "test error"),
                });
            }

            let (ny, nx) = self.shape;
            let mut array = Array2::from_elem((ny, nx), self.fill_value);

            if let Some(ref func) = self.value_generator {
                for j in 0..ny {
                    for i in 0..nx {
                        array[[j, i]] = func(i, j, nx, ny);
                    }
                }
            }

            Ok(array)
        }
    }

    #[test]
    fn test_mock_loader_returns_configured_shape() {
        let loader = MockRasterLoader::new()
            .shape((50, 20)) // ny=50, nx=20
            .fill_value(-10.0)
            .build();

        let result = loader.load_array("dummy_path.tif", (20, 50), None).unwrap();

        assert_eq!(result.dim(), (50, 20));
        assert_eq!(result[[0, 0]], -10.0);
        assert_eq!(result[[49, 19]], -10.0);
    }

    #[test]
    fn test_mock_loader_with_variable_shape() {
        let shapes = vec![(10, 10), (100, 50), (1, 1)];

        for (ny, nx) in shapes {
            let loader = MockRasterLoader::new().shape((ny, nx)).build();

            let result = loader.load_array("test.tif", (nx, ny), None).unwrap();
            assert_eq!(result.dim(), (ny, nx));
        }
    }

    #[test]
    fn test_mock_loader_error_handling() {
        let loader = MockRasterLoader::new().should_fail(true).build();

        let result = loader.load_array("test.tif", (10, 10), None);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), MhError::Io { .. }));
    }

    #[test]
    fn test_mock_loader_with_gradient_field() {
        let loader = MockRasterLoader::new()
            .shape((100, 100))
            .with_generator(|i, _j, nx, _ny| {
                let denom = nx.saturating_sub(1).max(1) as f64;
                -10.0 + 20.0 * (i as f64 / denom)
            })
            .build();

        let result = loader.load_array("gradient.tif", (100, 100), None).unwrap();

        assert!(result[[0, 0]] < result[[0, 50]]);
        assert!(result[[0, 50]] < result[[0, 99]]);
    }
}

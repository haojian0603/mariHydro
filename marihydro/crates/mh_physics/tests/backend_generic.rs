// crates/mh_physics/tests/backend_generic.rs

//! Backend泛型化测试
//! 验证f32/f64后端的一致性和正确性

use mh_physics::core::{Backend, CpuBackend, MemoryLocation};

/// 测试f32/f64后端的一致性
#[test]
fn test_f32_f64_consistency() {
    let backend_f32 = CpuBackend::<f32>::default();
    let backend_f64 = CpuBackend::<f64>::default();
    
    let n = 1000;
    
    // 分配缓冲区
    let mut x_f32 = backend_f32.alloc_init(n, 1.0f32);
    let mut y_f32 = backend_f32.alloc_init(n, 2.0f32);
    let mut x_f64 = backend_f64.alloc_init(n, 1.0f64);
    let mut y_f64 = backend_f64.alloc_init(n, 2.0f64);
    
    // axpy: y = 0.5 * x + y
    backend_f32.axpy(0.5, &x_f32, &mut y_f32);
    backend_f64.axpy(0.5, &x_f64, &mut y_f64);
    
    // 比较结果
    for i in 0..n {
        let diff = (y_f32[i] as f64 - y_f64[i]).abs();
        assert!(diff < 1e-5, "f32/f64 inconsistency at index {}: diff = {}", i, diff);
    }
}

/// 测试dot产品精度
#[test]
fn test_dot_precision() {
    let backend = CpuBackend::<f64>::default();
    let n = 10000;
    
    let x = backend.alloc_init(n, 1.0);
    let y = backend.alloc_init(n, 1.0);
    
    let result = backend.dot(&x, &y);
    let expected = n as f64;
    
    assert!((result - expected).abs() < 1e-10, "Dot product error: {} vs {}", result, expected);
}

/// 测试reduce操作
#[test]
fn test_reduce_operations() {
    let backend = CpuBackend::<f64>::default();
    
    let mut data: Vec<f64> = (0..100).map(|i| i as f64).collect();
    
    let max = backend.reduce_max(&data);
    let min = backend.reduce_min(&data);
    let sum = backend.reduce_sum(&data);
    
    assert_eq!(max, 99.0);
    assert_eq!(min, 0.0);
    assert_eq!(sum, 4950.0); // 0 + 1 + ... + 99
}

/// 测试正性保持
#[test]
fn test_enforce_positivity() {
    let backend = CpuBackend::<f64>::default();
    
    let mut data = vec![-1.0, 0.0, 1.0, -0.5, 2.0];
    backend.enforce_positivity(&mut data, 0.0);
    
    assert!(data.iter().all(|&x| x >= 0.0));
}

/// 测试内存位置
#[test]
fn test_memory_location() {
    let backend = CpuBackend::<f64>::default();
    assert_eq!(backend.memory_location(), MemoryLocation::Host);
}

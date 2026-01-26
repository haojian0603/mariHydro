// crates/mh_runtime/benches/simd_benchmarks.rs

//! SIMD 性能基准测试
//!
//! 测试 SIMD 内核的性能与标量版本的对比
//!
//! # 运行
//!
//! ```bash
//! cargo bench --package mh_runtime --bench simd_benchmarks
//! ```

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};

// 模拟 SIMD 操作（实际运行需要链接到 mh_runtime）
mod mock_simd {
    /// 标量版本的通量计算
    #[allow(unused_variables)]
    pub fn compute_flux_scalar(
        h: &[f64],
        hu: &[f64],
        hv: &[f64],
        _z: &[f64],
        g: f64,
        flux_h: &mut [f64],
        flux_hu: &mut [f64],
        flux_hv: &mut [f64],
    ) {
        let n = h.len();
        for i in 0..n {
            let h_val = h[i];
            if h_val > 1e-6 {
                let u = hu[i] / h_val;
                let v = hv[i] / h_val;
                flux_h[i] = hu[i];
                flux_hu[i] = hu[i] * u + 0.5 * g * h_val * h_val;
                flux_hv[i] = hu[i] * v;
            } else {
                flux_h[i] = 0.0;
                flux_hu[i] = 0.0;
                flux_hv[i] = 0.0;
            }
        }
    }

    /// AVX2 优化版本（模拟）
    #[cfg(target_arch = "x86_64")]
    pub fn compute_flux_avx2(
        h: &[f64],
        hu: &[f64],
        hv: &[f64],
        z: &[f64],
        g: f64,
        flux_h: &mut [f64],
        flux_hu: &mut [f64],
        flux_hv: &mut [f64],
    ) {
        #[cfg(target_feature = "avx2")]
        unsafe {
            use std::arch::x86_64::*;
            
            let n = h.len();
            let chunks = n / 4;
            let remainder = n % 4;
            
            let g_vec = _mm256_set1_pd(g);
            let half = _mm256_set1_pd(0.5);
            let eps = _mm256_set1_pd(1e-6);
            
            for i in 0..chunks {
                let offset = i * 4;
                
                let h_vec = _mm256_loadu_pd(h.as_ptr().add(offset));
                let hu_vec = _mm256_loadu_pd(hu.as_ptr().add(offset));
                let hv_vec = _mm256_loadu_pd(hv.as_ptr().add(offset));
                let z_vec = _mm256_loadu_pd(z.as_ptr().add(offset));
                
                // 检查深度
                let mask = _mm256_cmp_pd(h_vec, eps, _CMP_GT_OQ);
                
                // 计算速度
                let u_vec = _mm256_div_pd(hu_vec, h_vec);
                let v_vec = _mm256_div_pd(hv_vec, h_vec);
                
                // 计算通量
                let h2 = _mm256_mul_pd(h_vec, h_vec);
                let gh2_half = _mm256_mul_pd(_mm256_mul_pd(g_vec, h2), half);
                
                let flux_h_vec = hu_vec;
                let flux_hu_vec = _mm256_add_pd(_mm256_mul_pd(hu_vec, u_vec), gh2_half);
                let flux_hv_vec = _mm256_mul_pd(hu_vec, v_vec);
                
                // 应用掩码
                let zero = _mm256_setzero_pd();
                let flux_h_masked = _mm256_blendv_pd(zero, flux_h_vec, mask);
                let flux_hu_masked = _mm256_blendv_pd(zero, flux_hu_vec, mask);
                let flux_hv_masked = _mm256_blendv_pd(zero, flux_hv_vec, mask);
                
                // 存储结果
                _mm256_storeu_pd(flux_h.as_mut_ptr().add(offset), flux_h_masked);
                _mm256_storeu_pd(flux_hu.as_mut_ptr().add(offset), flux_hu_masked);
                _mm256_storeu_pd(flux_hv.as_mut_ptr().add(offset), flux_hv_masked);
            }
            
            // 处理余数
            if remainder > 0 {
                let start = chunks * 4;
                compute_flux_scalar(
                    &h[start..],
                    &hu[start..],
                    &hv[start..],
                    &z[start..],
                    g,
                    &mut flux_h[start..],
                    &mut flux_hu[start..],
                    &mut flux_hv[start..],
                );
            }
        }
        
        #[cfg(not(target_feature = "avx2"))]
        {
            compute_flux_scalar(h, hu, hv, z, g, flux_h, flux_hu, flux_hv);
        }
    }

    /// 状态更新（欧拉步）
    pub fn update_state_scalar(
        h: &mut [f64],
        hu: &mut [f64],
        hv: &mut [f64],
        dh: &[f64],
        dhu: &[f64],
        dhv: &[f64],
        dt: f64,
    ) {
        for i in 0..h.len() {
            h[i] += dt * dh[i];
            hu[i] += dt * dhu[i];
            hv[i] += dt * dhv[i];
        }
    }

    /// 最大波速计算
    pub fn max_wave_speed_scalar(h: &[f64], hu: &[f64], hv: &[f64], g: f64) -> f64 {
        let mut max_speed = 0.0f64;
        
        for i in 0..h.len() {
            let h_val = h[i];
            if h_val > 1e-6 {
                let u = hu[i] / h_val;
                let v = hv[i] / h_val;
                let c = (g * h_val).sqrt();
                let speed_x = u.abs() + c;
                let speed_y = v.abs() + c;
                max_speed = max_speed.max(speed_x).max(speed_y);
            }
        }
        
        max_speed
    }
}

/// 生成测试数据
fn generate_test_data(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let h: Vec<f64> = (0..n).map(|i| 1.0 + 0.5 * ((i as f64) * 0.01).sin()).collect();
    let hu: Vec<f64> = (0..n).map(|i| 0.5 * ((i as f64) * 0.02).cos()).collect();
    let hv: Vec<f64> = (0..n).map(|i| 0.3 * ((i as f64) * 0.015).sin()).collect();
    let z: Vec<f64> = (0..n).map(|i| 0.1 * ((i as f64) * 0.005).cos()).collect();
    (h, hu, hv, z)
}

fn bench_flux_computation(c: &mut Criterion) {
    let sizes = [1024, 4096, 16384, 65536, 262144];
    
    let mut group = c.benchmark_group("flux_computation");
    
    for &size in &sizes {
        let (h, hu, hv, z) = generate_test_data(size);
        let mut flux_h = vec![0.0; size];
        let mut flux_hu = vec![0.0; size];
        let mut flux_hv = vec![0.0; size];
        let g = 9.81;
        
        group.throughput(Throughput::Elements(size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("scalar", size),
            &size,
            |b, _| {
                b.iter(|| {
                    mock_simd::compute_flux_scalar(
                        black_box(&h),
                        black_box(&hu),
                        black_box(&hv),
                        black_box(&z),
                        black_box(g),
                        black_box(&mut flux_h),
                        black_box(&mut flux_hu),
                        black_box(&mut flux_hv),
                    )
                })
            },
        );

        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(
            BenchmarkId::new("avx2", size),
            &size,
            |b, _| {
                b.iter(|| {
                    mock_simd::compute_flux_avx2(
                        black_box(&h),
                        black_box(&hu),
                        black_box(&hv),
                        black_box(&z),
                        black_box(g),
                        black_box(&mut flux_h),
                        black_box(&mut flux_hu),
                        black_box(&mut flux_hv),
                    )
                })
            },
        );
    }
    
    group.finish();
}

fn bench_state_update(c: &mut Criterion) {
    let sizes = [1024, 4096, 16384, 65536];
    
    let mut group = c.benchmark_group("state_update");
    
    for &size in &sizes {
        let (h, hu, hv, _) = generate_test_data(size);
        let dh: Vec<f64> = (0..size).map(|i| 0.001 * (i as f64 * 0.01).sin()).collect();
        let dhu: Vec<f64> = (0..size).map(|i| 0.002 * (i as f64 * 0.01).cos()).collect();
        let dhv: Vec<f64> = (0..size).map(|i| 0.0015 * (i as f64 * 0.01).sin()).collect();
        let dt = 0.001;
        
        let mut h_work = h.clone();
        let mut hu_work = hu.clone();
        let mut hv_work = hv.clone();
        
        group.throughput(Throughput::Elements(size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("euler", size),
            &size,
            |b, _| {
                b.iter(|| {
                    h_work.copy_from_slice(&h);
                    hu_work.copy_from_slice(&hu);
                    hv_work.copy_from_slice(&hv);
                    
                    mock_simd::update_state_scalar(
                        black_box(&mut h_work),
                        black_box(&mut hu_work),
                        black_box(&mut hv_work),
                        black_box(&dh),
                        black_box(&dhu),
                        black_box(&dhv),
                        black_box(dt),
                    )
                })
            },
        );
    }
    
    group.finish();
}

fn bench_max_wave_speed(c: &mut Criterion) {
    let sizes = [1024, 4096, 16384, 65536];
    
    let mut group = c.benchmark_group("max_wave_speed");
    
    for &size in &sizes {
        let (h, hu, hv, _) = generate_test_data(size);
        let g = 9.81;
        
        group.throughput(Throughput::Elements(size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("reduce", size),
            &size,
            |b, _| {
                b.iter(|| {
                    mock_simd::max_wave_speed_scalar(
                        black_box(&h),
                        black_box(&hu),
                        black_box(&hv),
                        black_box(g),
                    )
                })
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_flux_computation,
    bench_state_update,
    bench_max_wave_speed,
);

criterion_main!(benches);

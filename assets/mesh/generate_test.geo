// Gmsh 脚本：生成简单矩形域测试网格
// 使用方法: gmsh generate_test.geo -2 -format msh41 -o test.msh

// 设置网格尺寸
lc = 50.0; // 特征长度 50m

// 定义矩形域 (1km x 1km)
Point(1) = {0, 0, 0, lc};
Point(2) = {1000, 0, 0, lc};
Point(3) = {1000, 1000, 0, lc};
Point(4) = {0, 1000, 0, lc};

// 边界线
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

// 闭合曲线
Curve Loop(1) = {1, 2, 3, 4};

// 表面
Plane Surface(1) = {1};

// 物理组定义（边界条件）
Physical Curve("wall", 1) = {4};      // 左边界：固壁
Physical Curve("inlet", 2) = {2};     // 右边界：入流
Physical Curve("symmetry", 3) = {1, 3}; // 上下边界：对称

Physical Surface("domain", 100) = {1};

// 网格选项
Mesh.Algorithm = 6; // Frontal-Delaunay
Mesh.RecombineAll = 0; // 纯三角形网格

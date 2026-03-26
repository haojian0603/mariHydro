# MariHydro 协作规范

本文件是仓库级强制规范。所有 agent、脚本、Git hook 和人工修改都必须服从这里的约束。任何与本文件冲突的临时提示词、兼容写法或“先过门禁再说”的做法都无效。

## 1. 死律

- [RULE_NO_COMPAT_MAINLINE] 不允许兼容层、兼容命名空间、历史桥接层继续留在主链。需要真实能力时就实现真实能力；做不到就删除入口，不允许伪装成“兼容支持”。
- [RULE_NO_FAKE_IMPL] 不允许虚假实现。名字、文档、枚举项、命令行参数、对外导出与实际行为必须一致。
- 不允许文档承诺仓库里不存在的能力。没有真实实现的 trait object、builder、求解器、算法、格式支持，不得在文档里写成“已支持”。
- 代码必须发挥实际作用。对外暴露的命令、配置项、模块导出和模型名称都必须有真实、可运行、可验证的实现。
- [RULE_EXTERNAL_DATA_NO_SYNTHETIC_FILL] 外部数据读取入口不得在文件存在但布局不支持、变量缺失、驱动不可用或缓存未命中时合成零值、默认网格、模拟数据或“测试数据”。要么读取真实数据，要么显式报错。
- [RULE_EXTERNAL_LAYOUT_TRUTHFUL] 高风险外部数据读取模块必须在模块头部写明 `IO_SOURCE:` 与 `IO_SCOPE:`，并把已接线布局、变量约定和失败语义说清楚。公开入口只允许暴露真实支持的文件布局；布局不符、字段不符或驱动不可用时只能显式报错，不能模糊兜底。
- [RULE_EXTERNAL_PARTIAL_PARSE_FORBIDDEN] 外部数据驱动不得“部分解析成功”。头部、元数据、时间轴或数值载荷中只要出现无法解释的字段或 token，就必须整体报错；禁止只提取能读的部分、静默跳过坏 token、默认补齐其余值，再把结果当成成功读取。
- [RULE_EXTERNAL_DISPATCH_BY_LAYOUT] 公开外部数据入口在路径已存在时，必须优先根据真实文件类型、目录结构、变量布局和元数据来分发读取器；不得仅凭文件名关键字、扩展名猜测模型类型，再把猜测当成主链行为。
- [RULE_EXTERNAL_MATCHED_LAYOUT_MUST_VALIDATE] 一旦目录扫描、文件名或上游元数据已经把某个外部文件识别为“受支持布局候选”，后续驱动打开、变量对校验、坐标网格校验和分潮去重都必须完整通过；任一环节失败都要立即报错，不得 `continue` 跳过坏文件后继续拼装部分结果。
- [RULE_EXTERNAL_CLI_FAILURE_CONTEXT_EXPLICIT] 外部 CLI 驱动只要已经进入真实工具调用阶段，就必须保留工具身份、调用阶段和操作系统错误上下文。不得把可执行文件缺失、进程启动失败或非零退出码统一压扁成空泛的 `NotAvailable`、`OpenFailed` 或 `ReadFailed` 字符串；错误里必须能看出是哪个工具、在哪个阶段、因为什么失败。
- [RULE_NO_TOLERANT_PUBLIC_PARSERS] 公开解析入口不得暴露 `*_or_default`、`*_or_zero`、`*_or_identity` 这类“失败时伪成功”的主链接口。对外部格式、时间、坐标和投影元数据，失败就返回错误；只有降级本身具有真实业务语义时，才允许显式命名的降级函数存在。
- [RULE_NO_PUBLIC_HEURISTIC_DETECTORS] 仅靠文件名、扩展名或关键字猜模型类型的逻辑，只能作为模块内部提示，不得作为公开 API、公开 trait 语义或对外承诺的“检测器”。公开入口必须以真实布局、目录结构和元数据校验为准。
- [RULE_EXTERNAL_METADATA_STRUCTURE_REQUIRED] 外部数据驱动必须把必须存在的结构段当成硬约束，例如 `bands` 数组、变量声明、属性赋值和时间元数据。缺失这些结构时只能报错，不能回退成空数组、空字符串、默认单元、默认波段数，或跳过坏行继续解析。
- [RULE_NETCDF_HEADER_VARIABLES_REQUIRED] `ncdump -h` 的 CLI 回退头解析必须至少拿到真实的 `variables:` 段和至少一个变量声明，不能靠空 `CliHeader`、默认结构体或“只有 dimensions 没有 variables”的半头信息继续冒充可读 NetCDF。
- [RULE_EXPORT_METADATA_SERIALIZATION_EXPLICIT] 导出链路里的元数据序列化不得伪装成功。像 `boundary_names`、字段名列表、属性清单这类会进入 VTU/PVD/检查点/项目文件的元数据，只要序列化失败就必须显式报错并终止写出，不能回退成 `"[]"`、`""`、空对象或其他合成占位值。
- [RULE_METADATA_TIMESTAMPS_EXPLICIT] 检查点、快照、项目文件和其他持久化元数据里的 `created_at`、时间戳或生成时刻字段不得在系统时钟异常时回退成 `0`、Unix 纪元或其他合成占位值。要么显式失败，要么把“未知时间”编码成真实的可区分状态，不能把假时间戳写进产物。
- [RULE_CHECKPOINT_METADATA_EXPLICIT] 检查点文件里的 `config_hash`、`mesh_hash`、目录扫描结果和头部摘要不得使用 `0`、`u32::MAX`、静默跳过坏文件等哨兵语义伪装“缺失”或“可继续”。缺失元数据必须编码成真实的可区分状态，严格校验时必须显式报错；检查点目录里一旦存在损坏或结构错误的 `.mhck` 文件，也必须立即失败，不能在列举和清理阶段偷偷忽略。
- [RULE_SNAPSHOT_BOUNDARY_METADATA_EXPLICIT] 网格快照里的边界元数据不得把缺失边界 ID 写成 `u32::MAX`、`-1`、空字符串或其他导出层哨兵值。冻结网格缺少边界 ID 时必须在快照构建阶段显式失败；VTU、检查点和其他导出入口也必须先验证快照，再决定是否写出边界字段。
- [RULE_PIPELINE_SHUTDOWN_EXPLICIT] IO 管道的 `flush`、等待完成、优雅关闭、立即关闭和析构收尾都不得用 `bool`、`let _ = ...`、静默 `join` 失败或“只打印 warning 仍算成功”的方式掩盖失败。只要关闭请求未送达、队列未按时排空、工作线程 panic、统计锁损坏或超时阈值被突破，就必须在公开入口返回显式错误；析构阶段无法返回错误时，也必须留下明确日志，不能伪装成正常收尾。
- [RULE_IMPORT_GEOMETRY_STRUCTURE_REQUIRED] 外部矢量导入不得在 Polygon 或 MultiPolygon 缺少外环、环点数不足、线性环未闭合时继续返回空外环或部分几何。几何结构不完整就必须显式报错，不能把坏输入折成“空面”“空洞列表”或其他伪成功结果。
- [RULE_IMPORT_SEMANTIC_NAMES_EXPLICIT] GeoJSON 、边界条件、分区或其他带语义名称的导入要素不得在缺少 `name` 或等价语义标识时自动合成 `unnamed`、`zone` 等占位名称。一旦该要素已被识别为边界条件、分区、规划单元或其他业务实体，就必须显式报错，不能用合成名称伪装语义存在。
- [RULE_MULTIPART_SEMANTIC_NAMES_PRESERVED] 一旦外部语义要素已经声明了真实名称，就不得在拆分 MultiPolygon、MultiLineString 或其他 multipart 结构时合成 `name_1`、`name_2` 这类后缀名称。若业务确实需要区分分片，必须保留原始名称，并通过显式 `part_index`、`segment_index` 或等价结构另行表达分片身份。
- [RULE_IMPORT_FEATURE_ID_EXPLICIT] 外部 GeoJSON `Feature.id` 一旦存在，就必须按 RFC 7946 可接受的真实语义显式保留。字符串和数值 ID 必须原样读入并传递；布尔值、对象、数组或其他非法 ID 类型必须立刻报错。禁止把非法 ID 吞掉、回退成空串，或在顶层 `Feature` 与 `FeatureCollection` 两条路径上丢弃 `id`。
- [RULE_IMPORT_TABULAR_SKIP_INVALID_OPT_IN] 表格型导入不得把“跳过坏行”设成默认行为。`CsvConfig`、时序表格、边界强迫表格和类似入口必须默认严格；只有调用方显式开启 `skip_invalid` 或等价开关时，才允许跳过坏行继续解析。解析错误里的来源字段也必须保留真实文件路径或显式 `<string>`，不能回退成空串。
- [RULE_IMPORT_NULL_GEOMETRY_EXPLICIT] GeoJSON `Feature.geometry = null` 不得在导入阶段被静默丢弃、过滤或折叠成“空要素”。如果当前公开数据结构不能真实表达空几何，就必须在解析阶段显式报错，并把这种结构性失败保留到调用方。
- [RULE_EXTERNAL_SHAPE_METADATA_EXPLICIT] 外部数组、网格和变量的维度信息必须显式匹配。不得用 `unwrap_or_default()`、缺省 `0/1` 或隐式单例轴去猜测 shape；维度缺失、轴顺序不符或前导维长度不合法时只能报错。
- [RULE_EXTERNAL_DRIVER_INDEX_ACCESS_EXPLICIT] 外部数据驱动的公开访问器不得把越界索引、NoData 像元或维度不匹配折叠成 `Option::None`。像 `RasterBand::get`、`RasterBand::interpolate`、`Variable::get` 这类接口必须返回显式错误，并保留“越界”“NoData”“索引维度无效”等失败语义。
- [RULE_EXPORT_STATE_ACCESS_EXPLICIT] 导出链公开状态访问器不得把缺字段、索引越界或实现者内部失败折叠成 `Option::None`。像 `VtuState::scalar` 这类接口必须返回显式错误，并保留“字段缺失”“索引越界”等失败语义。
- [RULE_EXPORT_STATE_SHAPE_EXPLICIT] 导出链状态构造器不得接受长度不一致的数组切片并把失败拖到导出阶段。像 `SimpleState::new`、`StateWithScalars::new`、`with_scalar` 这类入口必须在构造期显式校验 shape，不允许后续靠 panic 或越界访问暴露错误。
- [RULE_GEO_PROJECTION_ERRORS_EXPLICIT] 地理投影主链上的辅助量计算（比例因子、收敛角、瓦片边界等）不得用 `NaN`、`0`、`(0,0)` 之类的数值哨兵伪装失败。若计算依赖可失败的正反投影步骤，公开辅助函数就必须返回错误；若理论上不应失败，则必须把“不可能失败”的前提写清楚，而不是留静默回退。
- [RULE_GEO_CONVERGENCE_REQUIRES_PROJECTED_TARGET] 收敛角和基于收敛角的矢量旋转补偿只对投影目标 CRS 有定义。目标 CRS 仍是地理坐标时，公开入口必须显式报错，不能返回 `0` 角度把“未定义”伪装成“无旋转”。
- [RULE_GEO_CONVERGENCE_EXACT_PROJECTION_FORMULA] 投影收敛角必须由目标投影本身给出真实语义：横轴墨卡托类投影走显式公式，Web Mercator 仅在定义域内按“经线保持竖直”的几何性质显式返回 `0`。主链禁止再用 `delta_lat`、有限差分北向量或其他近似扰动去推收敛角。
- [RULE_SPATIAL_RADIUS_QUERY_EXACT] 空间索引的半径查询必须按“候选包围盒 + 精确距离过滤”实现，并且对负半径、非有限半径显式报错。禁止依赖 `nearest_neighbor_iter(...).take_while(...)` 这类顺序副作用，把最近邻遍历伪装成真实范围查询。
- [RULE_MESH_CIRCLE_QUERY_EXACT] 网格空间索引的圆形查询必须按真实的“单元多边形与圆相交”语义实现，并且对非有限圆心、非有限半径和负半径显式报错。禁止再用包围盒中心点、代表点或“命中一些单元即可”的启发式近似冒充精确查询。
- [RULE_STRUCTURED_BED_ELEVATION_EXPLICIT] 结构化网格不得把缺失的床面高程解释成 `0.0` 或默认平床。`StructuredMesh::freeze` 只能冻结已显式提供的 `bed_elevation` 或显式声明的 `uniform_bed_elevation`；缺失时必须立即报错，不能伪造平床地形。
- [RULE_GEO_GEODESIC_FAILURES_EXPLICIT] 椭球测地线计算不得把非收敛、奇异点或算法失效折叠成 `Option::None`、`NaN` 或其他无语义哨兵。像 Vincenty 这类迭代算法，一旦达到迭代上限仍未收敛，就必须返回显式错误并把失败原因保留给调用方；只有真正的成功路径才能返回距离数值。
- [RULE_GEO_AFFINE_INVERSE_EXPLICIT] 仿射变换的逆矩阵求解不得把奇异矩阵、不可逆矩阵或行列式退化状态折叠成 `Option::None`。`AffineTransform::inverse`、`apply_inverse` 这类公开接口必须返回显式错误，并保留“奇异变换不可逆”的语义，禁止让调用方通过空值猜测失败原因。
- [RULE_WEB_MERCATOR_DOMAIN_EXPLICIT] Web Mercator 公开辅助面（经纬度转投影、投影反算、分辨率、比例尺、瓦片坐标、瓦片边界）只接受有限且位于 EPSG:3857/4326 定义域内的输入。经纬度越界、投影坐标超出 extent、tile_size=0、DPI<=0 等情况必须显式报错，不得通过纬度裁剪、内部 `expect` 或返回默认数值伪装成功。
- [RULE_WEB_MERCATOR_TILE_INDEX_EXPLICIT] Web Mercator 的公开 tile 辅助面必须区分“tile 索引”和“tile 角点”语义。对外暴露的 `tile_to_*` 查询只接受合法 tile 索引区间 `[0, 2^zoom-1]`，越界时显式报错；仅内部角点换算才允许访问 `[0, 2^zoom]` 边界，禁止把越界 tile 直接折算成经纬度。
- [RULE_INTERNAL_INVARIANT_DEFAULTS_FORBIDDEN] 一旦前置校验已经把内部状态约束为“非空轴”“合法年内序号”“存在末端值”等不变量，后续代码就不得再用 `unwrap_or(...)`、默认月份、默认端点或其他合成值掩盖不变量破坏。要么在前置校验阶段返回错误，要么在不变量被破坏时显式 panic/报错。
- [RULE_RUNTIME_SYSTEM_PROBES_EXPLICIT] 运行时硬件和系统探测不得伪造常量结果。读取 `/proc`、`sysfs`、Win32 系统信息或线程拓扑失败时，要么显式报错，要么明确落到“未知”状态；不能把 `8GB/4GB`、空字符串、空 CPU 列表或 `0` 解析值当成真实探测结果。
- [RULE_RUNTIME_PARALLELISM_PROBES_EXPLICIT] 运行时并行度和物理核心探测不得把 OS 查询失败、`/proc/cpuinfo` 字段缺失或字段损坏静默折叠成 `1`、默认核心数或“看起来可用”的线程上限。线程数探测失败就显式失败；只有在元数据整体缺席且语义明确时，才允许落到带注释的保守估计。
- [RULE_RUNTIME_TOPOLOGY_DEFAULTS_FORBIDDEN] 运行时拓扑类型不得再暴露伪造的 `Default` 实现。`NumaTopology`、`NumaThreadPoolConfig` 这类对象如果依赖 OS 探测才能成立，就必须走显式 `detect()` 路径；禁止在 `default()` 里偷偷合成单节点、单核心或零内存拓扑。
- [RULE_RUNTIME_ALLOCATOR_NODE_QUERIES_EXPLICIT] 运行时分配器如果无法观测真实 NUMA 节点归属，就必须返回“未知”，不能把所有指针统一冒充成 `node 0` 或任意固定节点。`get_node()` 这类接口只能报告真实可验证的信息，不能拿固定值伪装亲和性。
- [RULE_NO_SILENT_NUMERIC_FALLBACK] 解析、投影、坐标转换、时间转换和外部驱动读取失败时，不得偷偷折成 `0`、`0.0`、`ZERO`、恒等结果或默认网格。若某条 API 的语义就是“失败时回零/回默认值”，名称必须明确写成 `*_or_zero`、`*_or_identity` 之类的真实名字。
- [RULE_NO_TEXT_CORRUPTION] 源码、脚本、配置和仓库规范文件不得包含乱码、替换字符 `U+FFFD`、Unicode 私有区字符，或明显的连续问号替换残留。中文文本若损坏，必须在同一批内修复，并由门禁阻止再次入库。
- [RULE_NO_MIXED_SCRIPT_GARBAGE] 中文注释、模块文档、脚本说明和仓库规范文件中不得混入欧元符号、西里尔字母、全角拉丁字母、带圈汉字等典型乱码脚本残片。出现这类字符默认按文本损坏处理，必须同批修复。
- [RULE_PHYSICS_PROVENANCE] 物理公式必须正确，且能核验来源。修改或新增物理模型时，必须能追溯到一手资料、标准教材、原始论文、技术手册或行业标准；没有来源就不能提交。
- [RULE_PHYSICS_TAG_REQUIRED] 主链中的物理模型文件必须在模块头部写明 `PHYSICS_SOURCE:` 与 `PHYSICS_SCOPE:` 标签。`PHYSICS_SOURCE:` 至少给出作者/机构、年份、标题或 DOI 中的两项；`PHYSICS_SCOPE:` 必须写清适用范围、关键假设和本实现没有覆盖的部分。修改公式时必须同步更新这些标签。
- [RULE_STANDARD_FORMULA_OR_REMOVE] 任何命名标准公式只要缺少必要输入、必要参数或必要边界条件，就必须从主链删除；不得用固定默认值、球形假设、线性插值过渡或其他偷换公式的方式伪装完整实现。
- 任何这次被修改且仍留在主链公开面的高风险物理文件，都应优先补齐 `PHYSICS_SOURCE:` 与 `PHYSICS_SCOPE:` 标签；如果暂时连来源都说不清，就不要扩大它的导出面。
- 命名必须真实。若实现的是经验近似、启发式估计、实验模块，名称和注释必须明确写出真实身份，不得借用成熟模型名。
- 不允许把“简化版”“近似版”“实验版”公式挂在主链公开能力上却继续沿用成熟模型名。要么补齐真实公式，要么重命名并收缩导出面。

## 2. 公式与物理模型规则

- 对任何新增或修改的物理公式，必须在相邻注释、模块文档或同文件说明中写清楚：公式名称、来源、适用范围、主要假设。
- 来源说明至少要包含可追溯信息中的两项：作者/机构、年份、文献标题、标准号、教材名、论文 DOI、技术手册名称。
- 如果仓库里出现某个成熟模型名，但实现与标准公式不一致，处理方式只有两种：
  - 按标准公式补齐真实实现。
  - 删除该名称并改成真实描述。
- 若公式所需输入尚未进入当前状态或配置结构（例如粒形、圆度、附加实验校准量），只能先补齐输入再实现；在此之前不得把该公式保留在主链公开表面。
- 不允许出现“用 Manning 暂时代替 White-Colebrook”这类借名实现。
- 不允许出现“Natural Neighbor”这类对外名称，但内部只是距离加权或其他近似方法。
- 对无法在当前批次完成的物理模型，不得留下假入口；可以删除入口，或明确收缩为内部实验代码且不对主链导出。
- 经验公式可以使用，但必须明确写出其经验性质、校准条件和适用区间；不能把经验式写成普适理论公式。
- 若公式依赖经验系数、阈值、裁剪上限或正则化常数，必须说明物理意义或数值稳定性目的；不得只留下裸魔法数字。
- 遥感反演、观测算子和传感器代理模型不得内置无来源的默认系数。若关系依赖场景标定，就必须要求调用方显式提供标定参数，不能在主链偷偷塞入“通用经验值”。
- 水动力、波浪、植被、堰流、泥沙等工程经验系数同样适用这一条：没有可核验来源和适用边界，就不能以 preset、默认系数、典型场景快捷函数的形式进入主链。
- [RULE_FORMULA_NAME_HONEST] 真实公式名称只能用于真实实现。若实现的是“某个标准频谱 + 另一个显式方向散布近似”之类的组合，就必须把组合关系和近似类型写进函数名、注释和文档，不能简称为完整成熟模型。
- [RULE_PUBLIC_SURFACE_TRUTHFUL] 公开枚举、配置项、CLI 选项、模型类型和错误分支必须与真实可运行能力一一对应。若主链当前只有一种实现，就不允许继续暴露多种未接线选项，也不允许依赖“Unsupported*”运行时拒绝来伪装支持面。
- [RULE_CALIBRATED_AI_NAMING] 观测算子、遥感反演和 AI 经验关系如果依赖场景标定或经验拟合，必须在对外类型名、文档和错误说明中显式写出 `Calibrated` 或 `Empirical` 属性；不得直接冒充成完整物理传感器模型、标准观测算子或通用反演公式。
- [RULE_AI_STATE_CONTRACTS_EXPLICIT] AI 模型的训练就绪状态、序列化状态、归一化参数和输出尺寸必须显式校验。不得在模型未训练、状态损坏、保存失败、加载失败、归一化缺参或预测尺寸与物理状态不匹配时继续给出零填充、默认置信度或部分 apply 结果。
- [RULE_SOURCE_API_SEMANTICS_EXPLICIT] 水动力源项的公开 API 不得暴露未参与实际计算的参数，也不得在单元索引越界、缺失配置或衰减系数读取失败时静默回退成“忽略设置”、“无源项”或“单位因子”。不变量被破坏时必须显式 panic 或返回错误，不能伪装成合法物理状态。
- [RULE_NO_FAKE_BACKEND_SURFACE] 分支级不可用的后端能力不得继续保留公开模块、公开类型或状态枚举来冒充“已接入但当前不可用”。没有真实运行时，就删除公开入口，只保留底层抽象层对未来后端的中性扩展点。
- [RULE_SETTLING_FORMULA_INPUTS_COMPLETE] 沉降速度公式必须和输入集一致。像 Dietrich 这类依赖颗粒圆度、Corey 形状因子等额外输入的关系，在状态结构未显式建模这些参数之前不得进入主链导出、自动选择或默认配置。
- [RULE_DIFFUSION_TENSOR_PROJECTION_EXACT] 流向各向异性扩散必须按张量投影计算法向有效系数，主链只接受 `D_n = D_L cos²θ + D_T sin²θ` 这类二次投影关系；不得用 `D_L |cosθ| + D_T sinθ` 之类线性 surrogate 冒充张量投影。

## 3. CLI、配置与应用层规则

- CLI 可以做运行时精度分发，但必须调用真实求解器路径，不得通过假的 `SolverBuilder`、假的 `DynSolver` 包装层或空壳状态对象模拟运行。
- `--config`、`--mesh`、输出目录、输出间隔等公开参数，必须真正生效；暂不支持的选项必须显式报错，不能静默忽略。
- 配置验证必须基于真实配置结构，不允许手写一套过时字段检查冒充真实 schema。
- 若某个入口当前仅支持部分能力，必须在运行时明确拒绝未接入的能力，而不是静默降级或偷偷忽略。
- 外部数据读取模块若只支持特定文件布局、特定变量命名或特定目录组织方式，必须把这些约束写进模块文档和公开错误路径；不能依赖“自动猜测 + 失败后回退另一条路”来伪装支持面。

## 4. 架构与实现规则

- `mh_runtime` / `mh_physics` 的配置标量转换不允许静默回退；失败就显式报错或 panic，并带上下文。
- `mh_physics` 主链不允许新增 `legacy_*`、`compat_*`、`sources::legacy`、`legacy_limiters` 等桥接层。
- `mh_agent`、`mh_physics`、`mh_terrain` 中不允许继续扩散裸 `Vec<f64>`、裸索引和裸几何数组作为主链数据交换方式；若暂时保留，必须限制在配置层、统计层或导出层。
- 任何“简化实现”“占位实现”“临时实现”如果进入主链导出、CLI、公开模块或默认路径，都视为违规。

## 5. 提交与验证规则

- 每批改动至少覆盖一组完整问题，且不少于 7 个文件，再提交 Git。
- 每一批提交前必须通过：
  - `cargo check --workspace`
  - `cargo clippy --workspace --all-targets`
  - `cargo test --workspace`
  - `powershell -ExecutionPolicy Bypass -File scripts/check_tracked_temp_artifacts.ps1`
  - `powershell -ExecutionPolicy Bypass -File scripts/check_repo_contracts.ps1`
  - `powershell -ExecutionPolicy Bypass -File scripts/check_real_implementation_contracts.ps1`
  - `powershell -ExecutionPolicy Bypass -File scripts/verify_architecture.ps1`
  - `powershell -ExecutionPolicy Bypass -File scripts/architecture_audit.ps1`
- [RULE_GATES_STRICTER_ONLY] 门禁只能更严格，不能放水。任何脚本修改都必须说明拦截了什么新问题，不能通过删除检查来“让门禁变绿”。
- 门禁必须按路径和语义收严，不能用会误伤已核验真实公式的宽泛关键词黑名单替代判断。对真实公式的拦截只能针对误导命名、缺失来源说明或错误实现，不能因为公式名本身出现在源码里就一刀切。
- Git hook 是强制门禁的一部分：`pre-commit` 跑快速检查，`pre-push` 跑完整检查。任何 agent 都不得绕过 hook。

## 6. 文档与记录规则

- 规范变化先写 `AGENTS.md`，再同步到脚本门禁；不能只改文档不改门禁。
- 修复记录以仓库内跟踪文件和 Git 提交为准，不要求中途人工口头确认。
- 注释和文档字符串优先使用中文，且必须表达真实含义；不能写带误导性的宣传句式。
- 若某能力被删除，相关文档、示例、命令帮助、导出说明必须在同一批内同步删除或改写。

## 7. 编码与写入规则

- 含中文文件默认使用 UTF-8（无 BOM）。
- 修改中文注释、中文文档和脚本说明时，必须回读校验，确认没有写成乱码或问号替换。
- 不允许用会把中文写坏的 shell 重定向方式批量覆盖源文件。

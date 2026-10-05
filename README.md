# tight_binding

一个用于构建和计算紧束缚晶格模型的 C++ 研究项目。代码以晶胞和跃迁项定义模型，可生成动量空间及有限晶格的哈密顿量，并包含能带、态密度、格林函数和电导率计算示例。项目还提供可选的 Python 绑定和 CUDA 数值代码。

## 代码结构

| 文件 | 主要内容 |
| --- | --- |
| `tight_binding/unit_cell.h` | `UnitCell<Dim>`：晶格矢量、原子位置、跃迁项、动量空间哈密顿量与能带接口；支持 1D、2D、3D。 |
| `tight_binding/lattice.h` | `PartialLattice<Dim>`：有限晶格、稠密与稀疏哈密顿量、位置和速度算符、格林函数。 |
| `tight_binding/models.h` | Chain、Square、Honeycomb、graphene、Lieb、Kagome、FCC 等预设模型。 |
| `tight_binding/numerical.h`、`functions.h`、`evaluate.h` | 数值计算及态密度相关函数。 |
| `tight_binding/snippets.h` | 能带、态密度、电导率等实验和绘图示例。 |
| `tight_binding/main.cpp` | C++ 控制台程序入口。 |
| `tight_binding/py_lib.cpp` | 基于 pybind11 的 Python 模块入口。 |
| `tight_binding/numerics_cuda.cu` | CUDA 数值实现。 |

## 构建环境

仓库提供 Visual Studio 解决方案 `tight_binding.sln`，工程使用 MSVC `v143` 和 Windows 10 SDK。x64 配置设为 `stdcpplatest`；源码使用 `<print>` 等较新的标准库功能，因此需要支持这些功能的 Visual Studio 工具链。

源码直接引用 Eigen 和 FFTW 头文件。绘图示例使用项目中的 `pybind_plot.h`，并通过 Python 的 Matplotlib 绘图，因此还需配置相应的 Python、pybind11 和 Matplotlib 环境。Release x64 配置引用 Intel oneMKL；工程文件还导入 CUDA 12.4 的 Visual Studio 构建规则，Release x64 链接 CUDA 库。所需头文件、库路径和工具集应按本机安装位置在 Visual Studio 中检查及配置。

打开解决方案后，可选择 `Debug|x64` 或 `Release|x64` 构建控制台程序。工程还定义了 `Release without CUDA|x64`、`test|x64` 和 `python_lib|x64` 等配置；其中 `python_lib|x64` 生成 `.pyd`，工程内的 Python 头文件和库路径指向开发者本机的 Python 3.11 安装目录，需要自行调整。`Release without CUDA` 虽未列出 CUDA 链接库，工程仍导入 CUDA 12.4 构建规则。

## 运行与修改示例

当前 `main.cpp` 只启用了 `test_conductivity_Kubo_xy()`，其他调用均已注释。编译并运行控制台程序会执行此示例；如需运行其他实验，可在 `main.cpp` 中选择 `snippets.h` 中的函数后重新编译。部分示例会显示绘图窗口，部分计算可能耗时较长。

在 C++ 中，典型流程是复制一个预设晶胞、加入跃迁项，再建立有限晶格并取得哈密顿量：

```cpp
auto cell = models::Square;
cell.add_hopping("A", 1_NN, 1.0);
PartialLattice<2> lattice{cell, LatticeVector<2>{20, 20}};
auto H = lattice.get_Hamiltonian_sparse<double>();
```

`1_NN` 是代码定义的最近邻标记；具体跃迁规则和可用接口见 `unit_cell.h`。Python 绑定在 `py_lib.cpp` 中导出 `UnitCell1D/2D/3D`、部分预设模型及绘图函数；该配置依赖本机 Python 构建环境，仓库未提供独立的 Python 安装脚本。

## 许可

见 `LICENSE.txt`。

# `txt2dxf.py` 脚本说明

该脚本用于将 [SpatialLM](https://github.com/iva-mzsun/SpatialLM) 模型预测的文本格式布局转换为 DXF (Drawing Exchange Format) 文件。转换后的 DXF 文件可以方便地在 CAD 软件（如 AutoCAD）中进行可视化、评测和进一步编辑。

## 功能

1.  **解析 SpatialLM 输出**: 读取并解析 SpatialLM 生成的文本文件，提取其中的墙体（`Wall`）几何信息。
2.  **二维平面转换**: 忽略原始数据中的Z轴坐标和高度信息，只处理墙体在 X-Y 平面上的二维投影。
3.  **房间多边形构建**:
    *   将离散的墙体线段进行连通性分析，识别出属于同一个房间（或闭合空间）的墙体组合。
    *   将连通的墙体线段连接成闭合的多边形，每个多边形代表一个房间。
4.  **DXF 文件生成**:
    *   将每个构建好的房间多边形转换为 DXF 文件中的 `LWPOLYLINE`（轻量多段线）实体。
    *   所有房间的多边形都会被写入同一个名为 `wall_poly` 的图层。

## 核心逻辑详解

从离散的墙体线段到闭合的房间多边形，脚本主要经过以下几个步骤：

1.  **墙体线段提取 (`parse_spatiallm_text`)**
    *   脚本首先读取输入的文本文件。
    *   利用 `spatiallm.layout.Layout` 类来解析文本内容。
    *   遍历所有解析出的 `Wall` 对象，提取其二维端点 `(ax, ay)` 和 `(bx, by)`，并为每个墙体创建一个 `WallSegment` 对象。

2.  **连通分量分析 (`ConnectedComponentAnalyzer`)**
    *   **目的**: 将相互连接的墙体线段分组。理论上，一个独立的房间或建筑由一组相互连接的墙体构成。
    *   **合并近点**: 由于模型预测可能存在微小误差，脚本首先会合并距离非常近的端点（在指定的 `tolerance` 范围内），将它们视为同一点。
    *   **并查集算法**: 使用并查集（Disjoint Set Union）算法来找出所有的连通分量。如果两个墙体线段共享一个（合并后的）端点，它们就被认为属于同一个连通分量。
    *   **输出**: 该步骤的输出是一个列表，其中每个元素都是一组相互连接的 `WallSegment` 对象，代表一个独立的几何结构。

3.  **多边形构建 (`PolygonBuilder` - 简化逻辑)**
    *   **目的**: 将每个连通分量内的墙体线段顶点连接成一个闭合的多边形。
    *   **顶点收集**: 对于每个连通分量，脚本会收集其中所有墙体线段的端点。
    *   **顶点去重**: 再次根据容差 `tolerance` 对所有顶点进行去重，确保每个顶点只出现一次。
    *   **凸包排序 (简化)**: 为了形成一个简单（不自交）的多边形，脚本采用了一种简化的排序方法：
        1.  计算所有顶点的几何中心。
        2.  根据每个顶点相对于几何中心的角度（使用 `arctan2`）进行排序。
        3.  这种方法适用于大多数凸多边形或近似凸多边形的房间形状。
    *   **输出**: 生成一个由有序顶点列表构成的多边形列表。

4.  **DXF 文件生成 (`DXFGenerator`)**
    *   **初始化**: 创建一个新的 DXF 文档，并创建一个名为 `wall_poly` 的图层。
    *   **实体创建**: 遍历上一步生成的所有多边形，为每个多边形创建一个 `LWPOLYLINE` 实体。
    *   **闭合与保存**: 将 `LWPOLYLINE` 的 `closed` 属性设置为 `True`，确保多边形是闭合的，并将其分配到 `wall_poly` 图层。最后，将所有内容保存到指定的 `.dxf` 文件中。

## 依赖

脚本运行需要以下 Python 库：

*   `ezdxf`: 用于创建和操作 DXF 文件。
*   `numpy`: 用于进行几何计算。

你可以通过 pip 安装这些依赖：

```bash
pip install ezdxf numpy
```

脚本在执行前会检查 `ezdxf` 是否已安装，如果缺失会给出提示。

## 使用方法

通过命令行运行此脚本。

### 命令格式

```bash
python tools/txt2dxf.py <input_file> [options]
```

### 参数说明

*   `input_file`: **必需参数**。指定输入的 SpatialLM 文本文件路径。
*   `-o, --output`: **可选参数**。指定输出的 DXF 文件路径。如果未提供，默认输出路径为 `<input_file_basename>.dxf`（例如，输入 `scene_01.txt`，则默认输出 `scene_01.dxf`）。
*   `--tolerance`: **可选参数**。设置点连接的容差范围，用于判断两个点是否为同一个点。默认值为 `1e-3`。
*   `-v, --verbose`: **可选参数**。启用详细模式，运行时会输出更详细的处理过程信息，方便调试。

### 示例

1.  **基本用法**:
    将 `scene_00000.txt` 转换为 `scene_00000.dxf`。

    ```bash
    python tools/txt2dxf.py path/to/scene_00000.txt
    ```

2.  **指定输出文件**:
    将 `input.txt` 转换为 `my_floorplan.dxf`。

    ```bash
    python tools/txt2dxf.py input.txt -o my_floorplan.dxf
    ```

3.  **调整容差并显示详细信息**:

    ```bash
    python tools/txt2dxf.py input.txt --tolerance 0.001 -v
    ```

## 输入文件格式

脚本期望的输入文件是 SpatialLM 的标准输出格式，其中包含对场景布局的描述。脚本主要关注 `Wall` 类型的定义。

示例如下：

```
wall_1 = Wall(ax, ay, az, bx, by, bz, height, thickness)
wall_2 = Wall(ax, ay, az, bx, by, bz, height, thickness)
...
door_1   = Door(wall_id, cx, cy, cz, width, height)
window_1 = Window(wall_id, cx, cy, cz, width, height)
```

脚本会解析 `Wall` 定义中的 `(ax, ay)` 和 `(bx, by)` 作为墙体线段的两个端点。`Door` 和 `Window` 的信息在当前版本中会被忽略。
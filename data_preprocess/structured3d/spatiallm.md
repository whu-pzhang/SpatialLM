SpatialLM 把结构信息当成“可读可编辑的 Python 脚本”，用 3 个 `dataclass` 表示墙、门、窗
- **Wall**
    - `a_x, a_y, a_z`：**墙底边**一端点 A 的 3D 坐标（整数）
    - `b_x, b_y, b_z`：**墙底边**另一端点 B 的 3D 坐标（整数）
    - `height`：墙高（整数）  
        这些整数**不是**原始米制坐标，而是**量化后的离散坐标**（下文解释量化） 。  
	对大部分墙，`a_z` 和 `b_z` 应该相同。
    - `thickness`: 墙的厚度（整数），取0

- **Door**
    - `wall_id`：该门附着的墙在脚本里的标识（如 `wall_7`）
    - `position_x, position_y, position_z`：门的定位点（整数，通常取开口中心点，见下文转换约定）
    - `width, height`：门的宽/高（整数，量化后单位同上）    
- **Window**
	- 字段与 Door 相同（附着墙、中心位置、宽/高）。
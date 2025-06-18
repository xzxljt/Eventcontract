# 交易记录表格 UI/UX 优化方案

## 1. 设计理念

本次优化的核心目标是，在保持信息清晰易读的基础上，提升交易记录表格的视觉“高级感”和现代化水平。我们将摒弃传统的、信息密集的表格样式，采用一种更注重呼吸感、层次感和交互细节的设计。

*   **信息降噪**：通过引入可展开行，将主要信息和次要信息分离，减少初始视觉负担。
*   **精致的视觉提示**：用更微妙、更融入整体设计的方式来区分不同状态的行。
*   **现代化的排版**：优化字体、颜色和对齐，引导用户视线，提升数据可读性。
*   **平滑的交互**：通过流畅的动画增强用户操作的质感。

---

## 2. 核心变更：可展开的表格行

这是本次设计的核心。我们将修改表格的HTML结构和Vue的渲染逻辑，以实现可展开/折叠的行。

*   **默认行**：只显示最关键的交易信息，如：信号时间、方向、投资额、盈亏额、结果。
*   **展开行**：点击默认行后，下方会展开一个子行（`sub-row`），其中包含次要信息，如：置信度、入场价、出场价、价格变化、交易后余额。

这种设计极大地简化了表格的初始视图，使其在任何屏幕尺寸下都易于浏览，避免了横向滚动。

---

## 3. 视觉设计细则

### 3.1. 表格整体样式

*   **表头 (`<thead>`)**:
    *   `background-color`: `var(--table-header-bg)` (或在深色主题下为 `#2A344A`)。
    *   `border-bottom`: `2px solid var(--primary-color)`。
    *   `font-weight`: `600` (Semibold)。
    *   `text-transform`: `uppercase` (可选，增加专业感)。
    *   `letter-spacing`: `0.5px` (可选，增加精致感)。
*   **行 (`<tr>`)**:
    *   移除 `table-striped` 的斑马条纹，改为通过 `border-bottom: 1px solid var(--border-color);` 来分隔行，更加简约。
    *   增加 `transition: background-color 0.2s ease;` 以配合悬浮效果。
*   **单元格 (`<td>`, `<th>`)**:
    *   增大垂直内边距：`padding: 0.85rem 1rem;`。

### 3.2. 数据排版

*   **文本数据** (如信号时间, 方向): 左对齐 (`text-align: left;`)。
*   **数值数据** (如投资额, 盈亏): 右对齐 (`text-align: right;`)。这对于比较数值大小至关重要。

### 3.3. 状态视觉提示

替代原有的整行高亮 (`.table-success`, `.table-danger`)，我们采用更微妙的方式：

*   在每行的**第一个单元格**（信号时间）前，添加一个彩色的竖条。
*   这个竖条的颜色根据交易结果（胜利/失败）而变化。

```css
.trade-row td:first-child {
    position: relative;
    padding-left: 1.5rem; /* 为竖条留出空间 */
}

.trade-row td:first-child::before {
    content: '';
    position: absolute;
    left: 0.5rem;
    top: 50%;
    transform: translateY(-50%);
    width: 4px;
    height: 20px;
    border-radius: 2px;
    background-color: var(--text-secondary-color); /* 默认颜色 */
}

.trade-row.is-win td:first-child::before {
    background-color: var(--success-color);
}

.trade-row.is-loss td:first-child::before {
    background-color: var(--error-color);
}
```

### 3.4. 可展开行的样式

*   **展开的子行 (`sub-row`)**:
    *   背景色可以比普通行稍深或稍浅，以作区分。
    *   `padding`: `1rem`。
    *   内容使用 `grid` 或 `flex` 布局，以“标签-数值”对的形式展示次要信息。
*   **交互图标**:
    *   在默认行的最后一列，放置一个 `chevron-down` 图标，表示可展开。
    *   当行展开时，该图标平滑地旋转为 `chevron-up`。

---

## 4. 实现步骤概要

### 4.1. HTML (`frontend/index.html`)

1.  **修改 `<tbody>` 的 `v-for`**:
    *   将 `<tr>` 替换为 `<template>` 以包裹每个交易记录的多个行。
    *   创建一个主行 `<tr>` 用于显示核心数据，并添加一个 `@click` 事件来切换一个本地状态（如 `pred.isExpanded = !pred.isExpanded`）。
    *   创建一个子行 `<tr>`，使用 `v-if="pred.isExpanded"` 来控制其显示。这个子行的 `<td>` 将 `colspan` 设置为所有列的总数，内部再进行布局。
2.  **调整列**:
    *   主行只保留“信号时间”、“方向”、“投资额”、“盈亏额”、“结果”和“操作”列（用于放置展开图标）。
    *   将其余列的数据移至子行中显示。

### 4.2. JavaScript (`frontend/index-scripts.js`)

1.  **数据初始化**: 在 `runBacktest` 成功后，为返回的 `predictions` 数组中的每个对象添加一个 `isExpanded: false` 属性。
2.  **方法**: 不需要额外的方法，因为切换逻辑可以直接在 `@click` 中完成。

### 4.3. CSS (`frontend/static/css/index-styles.css`)

1.  **添加新样式**: 将上述 3.1, 3.2, 3.3, 3.4 中提到的所有新CSS规则添加到此文件中。
2.  **覆盖旧样式**: 移除或覆盖掉 `shared-styles.css` 中可能冲突的通用表格样式（特别是背景色相关的）。

这个方案在提升视觉效果的同时，极大地改善了信息架构和可用性，符合“高级而不过度复杂”的要求。
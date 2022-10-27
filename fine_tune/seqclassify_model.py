# -*- coding: utf-8 -*-

from torch import ne
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import XLNetTokenizer, XLNetModel, XLNetForSequenceClassification
import torch


'''
corpus = ('阿拉法特 阿拉伯 紧急 首脑会议 对 以色列 屠杀 事件 采取措施 内容 新华社 突尼斯 月 日电 在 以色列 士兵 星期日 屠杀 巴勒斯坦人 事件 发生 后 巴勒斯坦国 总统 阿拉法特 今天 即将 举行 阿拉伯 紧急 首脑会议 对此 事件 采取措施 阿拉法特 对 蒙特卡洛 电台 发表谈话 时 指出 阿拉伯人 忍耐 是 有 限度 他 紧急 首脑会议 采取 具体措施 而 不是 仅仅 通过 一些 声明 巴勒斯坦国 外交部长 法鲁克 卡杜米 今天 在 前往 巴格达 途经 罗马 时说 联合国 在 被占领土 上 派驻 观察员 巴勒斯坦 法塔赫 领导人 就 这次 屠杀 事件 对 巴林 天天 报 发表谈话 说 这 一 事件 会 导致 各种 可能性 在 被占领土 任何 力量 都 不能 阻止 巴勒斯坦人 采取任何 行动 解放 巴勒斯坦 人民阵线 总书记 哈巴 什 星期一 在 大马士革 举行 记者 招待会 立即 举行 巴勒斯坦全国委员会 会议 在 加强 巴勒斯坦 人民 起义 斗争 基础 上 制订 新 政策 以 迫使 以色列 接受 巴勒斯坦 完', '航空 工艺技术 年 第期 电子 经纬仪 测量 技术 徐枝 和 摘要 简述 电子 经纬仪 测量 系统 基本原理 总结 该 仪器 测量 技术 对 工装 设计 制造 分析 测量误差 原因 介绍 减小 误差 方法 同时 提出 今后 开发 应用 思路 电子 经纬仪 测量误差 开发 应用 电子 经纬仪 测量 系统 是 用 手工 光学 瞄准 非 接触 移动式 三维 测量 系统 用于 几何 量 测量 它 具有 许多 普通 三 坐标 测量机 所 不 具备 特点 测量范围 可 达 几十米 不 测量 工作台 与 导轨 只要 被 测量 物体 随意 放置 平稳 即可 便于 携带 与 安装 对 正在 作业 和 制造 中 物体 或 到 安装 现场 测量 同时 也 具有 简便 数据 自动 处理 等 特点 电子 经纬仪 测量 系统 用于 许多 领域 如 航空航天 造船 重型机械 制造 建筑业 等 它 能 迅速 准确 地 对 大型 及 不规则 物体 实地 三维 测量 和 计算 即使 对 形状 复杂 工件 及 难以接近 物体 测量 精度 也 达到 测量 系统 组成 及 基本原理 组成 电子 经纬仪 系统 是 由 台 或 多台 电子 经纬仪 传感器 计算机 及 相应 接口 通信 电缆 碳素纤维 亦 有 铝制 或 钢制 标尺 等 组成 仅 以台 电子 经纬仪 为例 条长 芯 电缆 分别 将 和 台 经纬仪 数据 传送 插座 与 多路 数据 采集器 个 相应 输入 插座 相连接 与 计算机 之间 由 连接 以 实现 台 经纬仪 与 计算机 之间 数据 与 指令 传输 并 向 台 经纬仪 提供 直流电源 至于 电子 经纬仪 数量 可 根据 被测 物体 情况 灵活 选用 台 组合 图 测量 系统 基本 组成 示意图 基本原理 坐标 测量 采用 前方 交汇 原理 操作员 至少 用台 经纬仪 瞄准 同 一点 测量 以台 电子 经纬仪 工作 系统 为例 其 原理 见 图图 中 利用 立体几何 边角 关系 和 三角函数 可 得出 下列 方程组 图 测量 基本原理 方程组 中为 所 测点 三维 坐标值 为 台 经纬仪 之间 距离 和 高度 差 其中 为 粗略 测量 值建站 时 将 数据 输入 计算机 为 台 经纬仪 互望 瞄准 时 角度 值 分别 为 台 经纬仪 在 瞄准 被 测点 时 所得 两个 方向 上 测量 角 上述 各种 数据 值均 由 计算机 处理 最终 得到 是 被 测点 值 要 想得到 在 经纬仪 坐标系 下 坐标值 即值 则 要 通过 系统 定 标的 过程 亦 即 经过 两 经纬仪 互 瞄 和 测量 一 已知 长度 碳素纤维 标尺 再 通过 空间 解析 计算 即可 得到 值 这些 计算 由 计算机 自动 完成 电子 经纬仪 系统 虽 是非 接触式 但 必须 通过 专用 测量 孔 或 粘贴 瞄准 靶标 来 指示 目标 亦可 采用 激光 打点 测量 测量 基本 过程 在 测量 之前 必须 根据 待测 物体 特点 将 经纬仪 设置 到 最佳 位置 并 调平 电子 调平 一般 控制 在 以内 水泡 调平 控制 在 一格 以内 建立 经纬仪 坐标系 驻地 坐标系 这 一步 至关重要 而且 是 难点 经纬仪 坐标系 建立 好坏 直接 影响 测量 精度 坐标 转换 即 物体 坐标系 建立 通过 测量 物体 上 至少 个 相当 准确 且 分布 合理 基准点 对于 大型 物体 则 更 多 基准点 把 驻地 坐标系 下 坐标值 转换 到 实际 测量 点 坐标值 在线 测量 物体 坐标系 建立 以后 就 开始 所有 测量 将 测量 数据 存入 计算机 内 数据处理 将 检测 数据 在 经纬仪 专用 软件 或 外部设备 中 处理 输入 中 与 理论 数模 比较 得出 其 与 数模 理论 型面 最小 法向 距离 测量 系统对 工装 设计 制造 电子 经纬仪 测量 系统 技术 特点 对 模具 夹具 设计 提出 下列 要 有 统一 制造 测量 基准 一般 是 个 或 个 以上 孔所 空间 基准 设计 时 可 根据 具体情况 这些 基准 孔 在 同一 平面 内 也 不 在 同一 平面 内 必须 给出 这些 孔 中心 三维 坐标值 才能 满足 测量 如果 制造 时 没有 保证 这些 数据 必须 给出 加工 实际 值 以便 测量 时 建立 物体 坐标系 否则 所 测量 数据 将 难以 与 理论 数模 比较 这 一点 是 十分 重要 夹具 设计 时 必须 给出 所有 测量点 三维 坐标值 并 通过 格式 编制 文本文件 能 利用 软盘 将 数据 输入 电子 经纬仪 测量 系统 计算机 内 便于 跟踪 测量 安装 模具设计 时 一般 只 给出 测量 基准点 三维 坐标值 控制 模具 型面 制造 质量 必要 时 设计 工艺 质保 测量 等 人员 要 共同 研究 测量点 分布 和 点数 测量 结束 后 将 测量 数据 输入 分析 提高 测量 精度 和 进度 模 夹具 制造 时 必须 保证 基准 一致性 即 设计 基准 加工 基准 测量 基准 是 同一 基准 一旦 不能 保证 时 必须 给出 实际 基准 数据 测量误差 分析 及 控制系统 在 定向 点 坐标 测量 时 误差 来源 有 两项 一是 经纬仪 系统误差 二是 测量 时 偶然误差 系统误差 是 指 经纬仪 本身 竖盘 倾斜 误差 指标 差 和 照准 差 等 而 这些 误差 通过 经纬仪 本身 程度 存储 补偿 予以 消除 系统误差 表现 在 测角 精度 误差 这 是因为 经纬仪 制造 时其 调平 转角 水平角 和 竖 直角 都 不可避免 地会 制造 精度 上 误差 误差 与 经纬仪 安装 位置 有 很大 关系 一是 与 被 测点 间 距离 有关 二是 与 经纬仪 测量 时 瞄准 交会 角 有关 图 显示 测角 误差 与 经纬仪 至 被 测点 间 距离 之间 关系 图 测量 距离 与 测角 误差 关系 从图 看出 测角 误差 随着 经纬仪 与 被 测点 距离 增加 而 加大 控制 测角 误差 为 最小 在 经纬仪 安装 时 必须 使 其 与 被 测点 之间 距离 在 保证 覆盖 被 测点 前提 下 尽可能 地小在 实际 测量 时 一般 控制 在 以内 那么 可 在 以内 电子 经纬仪 测量 时 交会 角对 被 测点 位 坐标 精度 关系 每台 经纬仪 在 瞄准 点 目标 时 都 会 有 一 发散 区域 当台 经纬仪 交会 时其 发散 区域 在 空间 表现 为 两个 近似于 圆锥体 相交 图 所示 是 误差 区域 在 经纬仪 和 被 测 点点 组成 平面 上 投影 两 经纬仪 之间 距离 为 经纬仪 转角 误差 为用 平面 坐标系 得到 条 直线 方程 分别 为 直线 直线 直线 直线 运用 直线 相交 求 交点 方法 可 求 出点 坐标值 点 所 组成 四边形 面积 为 当时 有 最小值 且 为 零 从 上述 分析 得出 两 经纬仪 空间 交会 角为 图 交会 角对 测点 坐标 精度 影响 图 发散 误差 在 平面 上 投影 时 测量误差 最小 所以 在 实际 瞄准 时应 注意 空间 交会 角对 测量 精度 影响 在 测量 时要 保证 每个 测量点 都 为 是 不 可能 根据 正弦曲线 特性 当 角度 为 时值 在 到 之间 变化 且 变化 缓慢 测量 时取 就 上述 两种 消除 或 减小 误差 方法 是 有 矛盾 并且 与 经纬仪 对 被 测点 覆盖面 有关 图中点 所 组成 四边形 为 所 需 测量点 分布 范围 其中 为 台 经纬仪 安装 位置 为 台 经纬仪 之间 距离 为 经纬仪 离 被 测 工件 距离 如何 设置 经纬仪 位置 即 确定 尺寸 对 提高 测量 精度 减小 误差 是 有 作用 图 经纬仪 位置 与 被 测 物体 关系 根据 习惯 一般 把 经纬仪 安装 在 被 测 物体 测量点 分布 范围 外即 有 在 实际 测量 中 特别 是 在 测量 大型 工件 时 往往 采用 多台 经纬仪 或 移动 经纬仪 以 保证 测量误差 最小 测量误差 有人 视觉 误差 环境 对 经纬仪 影响 如 振动 光线 等 所 造成 误差 另外 被测 物体 基准 孔 制造 误差 对 建立 坐标系 所 带来 误差 影响 最大 上述 因素 在 测量 过程 中 都 须 严格 加以控制 首先 向 计算机 中 输入 所 需 控制 精度 值如 测量 时若 超差 计算机屏幕 上 就 会 显红 此时 必须 采取措施 加以克服 测量 系统 开发利用 电子 经纬仪 测量 系统 经过 前 阶段 应用 虽然 取得 成果 对 和 两个 软件 本身 功能 来说 没有 得到 充分 应用 还有 一部分 功能 未 经过 实践 已经 应用 功能 有待于 巩固 提高 和 熟练 随着 技术 发展 将 有 更新 测量 软件 问世 如 激光 跟踪 式 运动 目标 三 坐标 测量 系统 下面 就 开发 应用 谈 几点 看法 激光 跟踪 测量 我 公司 现有 和 两种 软件 都 具有 激光 测量 功能 以及 数字 跟踪 功能 也 引进 激光 发射器 等 装置 如何 把 跟踪 测量 与 激光 测量 结合 起来 加以 应用 是 今后 开发 任务 之一 特别 是 已知 测量点 三维 坐标值 固定 其中 二维 坐标 求 第三 坐标值 误差 时 利用 该软件 方法 用于 大型 构件 如 飞机 桥梁 大型 设备 等 测量 上 是 十分 方便 同时 对 飞机 汽车 仿形 开发 设计 和 检测 也 是 十分 有效 此外 据 有关 资料 介绍 国外 已经 开发 和 应用 激光 跟踪 式 运动 目标 三 坐标 测量 系统 它 是 只用 传感器 便 能 对 运动 目标 高速 高精度 数据 采集 测量 系统 用于 测量 运动 物体 或 将 不规则 形状 物体 表面 数字化 整个 测量 系统 一人 操作 无需 人工 瞄准 数据 采集 速度 高 达 每秒 测量 个 点位 这一 技术 开创 新 测量 领域 例如 向 工业 机器人 输入 测试程序 后 就 能 快速 和 方便 地 检查和 改善 它 定位精度 另外 可用 本 测量 系统 无 接触 地 对 工业 机器人 多次 静态 测试 和 动态 特性 测试 隐藏 点 测量 精度 提高 隐藏 点 测量 是 电子 经纬仪 功能 所谓 隐藏 点 是 指 不能 被 任意 两台 电子 经纬仪 所 瞄准 点 即使 移动 经纬仪 也 不能 瞄准 此外 测量 时因 交会 角太 大将 会 影响 测量 精度 时 减小 测量误差 也 应 采用 隐藏 点 测量方法 以 保证 测量 精度 在 模具 测量 中 已经 应用 此 测量方法 但 在 应用 中 隐藏 杆 摆放 和 固定 不当 影响 测量 精度 今后 任务 是 提高 测量 技巧 设计 方便 固定 支座 使得 隐藏 杆 尖点 真正 对准 所 测点 从而 提高 测量 精度 测量 现场 坐标 转换 和 数据 比较 测量 数据 与 产品 理论 数模 比较 是 在 计算机 站 这样 只能 是 在 测量 结束 后 如果 制造 基准 或 测量 有误 在 现场 无法 判断 是 制造 基准 问题 还是 测量 问题 夹具 测量 基本上 不会 出现 此类 问题 只要 夹具 设计 时 直接 将 测量 坐标系 建成 产品 理论 坐标系 所 给 测量 值 是 在 产品 理论 坐标系 下 三维 坐标值 现场 就 数据 比较 模具 则 不然 很 容易 出现 上述 问题 模具 测量 是 在 模具 表面 贴 上 若干个 靶标 这 就是 测量点 这些 点 是 事先 不能 给定 只有 先对 模具 表面 上 点 测量 然后 再 把 测量 数据 输入 到 中 才能 比较 如果 在 模具 制造 车间 设立 一 测量 站 增添 一台 大型 三 坐标 测量机 或 一套 电子 经纬仪 测量 系统 同时 配备 一台 工作站 利用网络 将 模具设计 理论 数据 传递 到 工作站 再 将 测量 数据 也 随时 传到 工作站 就 可 实现 测量 数据 现场 分析 发现 问题 及时 解决 单位 昌河 飞机 工业 公司 研究员级 高级 工程师', '中国 环境 科学 年 第期 美国能源部 谈 石油 禁运 以来 能源 使用 状况 美国能源部 能源 信息 处 发布 一份 新 题为 年 阿拉伯 石油 禁运 周年 报告 该 报告 表明 年 来 世界 能源 有 很大 变化 但 并 不 总是 好消息 尽管 担心 对 国外 石油 过度 依赖 但 美国进口 石油 比重 仍 从 年 上升 到 年 总 能源 消费 增加 以上 而 人均 能源 用量 基本上 持平 报告 中说 按 计算 能源 消费 自年 以来 下降 约 这 是 生产 效率 提高 石油 发电 比例 从 急剧下降 到 这 是 更 多 煤炭 用于 发电 近年来 则 有 更 多 天然气 用于 发电 机动车 效率 提高 人均 石油 用量 从 年 以来 下降 过去 年 效率 下降 配备 大功率 发动机 轻型 卡车 销售 增加 与此同时 按 通货膨胀率 调整 每加仑 液化 天然气 价格 基本上 和 年 相同 即 比 年 高峰 时 跌落 江英 摘自', '出处 拉丁美洲 研究 原刊 地名 京原 刊期 号 原刊 页 号 分类号 分类 名 国际 政治 复印 中美洲 向 何处 去 中美洲 政治 现状 与 趋势 杨建民 世纪 年代 整个 中美洲地区 战火 不断 曾一度 成为 国际舆论 关注 热点 地区 中美洲 问题 已 基本 得到 解决 斗转星移 中美洲 现在 政治形势 如何 世纪 中美洲 政治 将 如何 演变 本文 试图 对此 作一 简要 分析 一连 年 烽火 过去 中美洲 战乱 表现 为 于个 国家 两种 类型 内战 一是 尼加拉瓜 内战 年 月 以 桑地诺 民族 解放阵线 为主 革命 力量 推翻 索摩查 独裁统治 建立 新政府 新政府 对内 实行 社会 改革 对外 与 苏联 古巴 等 社会主义 国家 建立 密切 关系 从而 引起 国内 顽固 势力 反扑 前 索摩查 政府 残余分子 率先 建立 反 政府 武装 在 美国 支持 下 发动 内战 此外 一些 曾 拥护 革命 但 与 桑解阵 有 政见 分歧 力量 也 拉 起 队伍 二是 萨尔瓦多 和 危地马拉 内战 与 革命 前 尼加拉瓜 一样 这 两个 国家 在政治上 长期 实行 军事 独裁 经济落后 社会 矛盾尖锐 年 月 萨尔瓦多 几支 游击队 联合 成立 法拉本多 马蒂 民族 解放阵线 发动 全国性 起义 多次 向 政府军 发动 大规模 进攻 控制 萨尔瓦多 国土 在 危地马拉 年 出现 第一支 游击队 开始 武装斗争 年 月 国内 几支 游击队 实现 联合 组成 危地马拉 全国 革命 联盟 向 政府军 发起 强大 攻势 控制 大片 地区 中美洲 爆发 革命 运动 和 游击战争 是 当地 社会 矛盾激化 但 在 当时 全球 两极 对峙 国际 环境 中 美国 这些 革命 是 苏联 和 古巴 渗透 对 美国 后院 造成 严重威胁 尤其 是 在 里根 上台 后 美国 在 反击 苏联 进攻 思想 指导 下 对 中美洲 事务 肆意 干涉 一是 大力 扶植 尼加拉瓜 反 政府 武装 向 桑解阵 政权 发动 进攻 并 对 桑解阵 施以 孤立 封锁 和 制裁 政策 二是 竭力支持 萨尔瓦多 和 危地马拉 两国政府 提供 大量 武器 和 经济援助 帮助 它们 镇压 游击队 三是 把 洪都拉斯 变成 干涉 中美洲 事务 军事基地 派驻 大量 军队 通过 军事演习 等 手段 训练 萨尔瓦多 和 危地马拉 政府军 和 尼加拉瓜 反 政府军 这样 美国 在 中美洲 发动 一场 旨在 推翻 尼加拉瓜 政府 镇压 萨尔瓦多 危地马拉 两 国 游击队 低 烈度 战争 而 当时 尼加拉瓜 政府 和 萨尔瓦多 危地马拉 两国 游击队 得到 苏联 和 古巴 支持 两大 集团 出于 自身 利益 考虑 而 明争暗斗 中美洲 战乱 迟迟 无法 得到 解决 如 在 年 尼加拉瓜 政府 曾 率先 提出 与 反 政府 武装 签署 停火协议 建议 但 因 美国 继续 在 军事 上 支持 反 政府 武装 而 使 谈判 破裂 尼加拉瓜 与 洪都拉斯 发生 边界 冲突 萨尔瓦多 冲突 双方 也 迟迟 不能 进行谈判 从而 使 中美洲 长期 处于 危机 状态 年代 末 两极 格局 发生 重大 变化 国际局势 趋向 缓和 美苏 逐渐 调整 各自 中美洲 政策 年 苏联 领导人 戈尔巴乔夫 在 古巴 宣布 反对 向 中美洲 输出 革命 和 反革命 年 布什 上台 后 美国 也 转而 采取 以 政治 和 外交 手段 解决 中美洲 问题 尼加拉瓜 于 年 月 举行 大选 得到 美国 大力支持 查莫罗 夫人 当选 为 总统 随即 桑解阵 反 政府 武装 和 查莫罗 夫人 代表 就 停火 和 解散 反 政府 武装 达成 协议 联合国 维和部队 也 进入 洪都拉斯 和 萨尔瓦多 监督 停火 并 建立 个 安全区 年 萨尔瓦多 右翼 政党 民族主义 共和 联盟 在 大选 中 获胜 大大 增强 工商 阶层 对 和平 进程 信心 此后 经过 多次 谈判 萨尔瓦多 政府 和 游击队 终于 在 年月日 签署 停火协议 结束 年 内战 危地马拉 内战 是 中美洲 最早 发生 最后 熄灭 一场 战火 实现 和平 道路 艰难曲折 早 在 年 危地马拉政府 就 与 游击队 全国 革命 联盟 开始 谈判 但 涉及 富有 阶级 利益 和 军人 权利 问题 和平谈判 一波三折 进展 缓慢 年 新政府 上台 后 出现 和平 曙光 当年 月 危地马拉 总统 秘密 会见 游击队 位 司令 月 双方 相继 宣布 停止 军事行动 经过 半年 多 谈判 双方 就 社会 经济 和 土地 军队 在 社会 中 作用 修改 宪法 和 选举法 实现 最后 停火 等 问题 达成协议 并 于 年月日 签署 永久 和平 协定 结束 这场 持续 年 内战 这 也 标志 着 中美洲 实现 全面 和平 战乱 给 中美洲 带来 巨大 灾难 危地马拉 内战 造成 万人 死亡 经济损失 达 数十亿美元 尼加拉瓜 和 萨尔瓦多 分别 有万人 和 万人 死 于 战乱 经济损失 分别 高 达 亿美元 和 亿美元 中美洲地区 共有 万 人口 而万人 生活 在 贫困 之中 其中 万人 处于 赤贫 状态 战争 给 社会 和 经济 带来 严重破坏 失业人数 剧增 大批 农民 流入 城市 很多 复员军人 和 前 游击队员 没有 得到 安置 等 问题 使 社会 政治 稳定 受到 严峻 挑战 二 当前 中美洲 政治形势 特点 年代 以来 中美洲 国家 在 政治 逐渐 稳定 情况 下 积极 推进 民主化 进程 其 政治形势 出现 以下 几个 特点 第一 各国 达成 和平 协议 虽然 几经 曲折 但 基本上 都 已 落实 中美洲 和平 得以 巩固 尼加拉瓜 问题 是 中美洲 实现 和平 关键 年 月 尼加拉瓜 查莫罗 夫人 当选 总统 后 美国 从 洪都拉斯 和 伯利兹 撤回 军事 人员 古巴 也 撤回 驻 尼加拉瓜 军事顾问 同年 月 查莫罗 夫人 宣布 裁减 和 改组 军队 主张 军队 非政治化 桑解阵 领导人 也 表示 该 阵线 将 成为 建设性 反对派 并 支持 新政府 一切 有利于 尼加拉瓜 人民 政策 月 日 尼加拉瓜 反前 政府 武装 总参谋部 宣布 解除武装 标志 着 尼加拉瓜 年 内战 结束 然而 国内 和平 并未 而 实现 武装斗争 仍 被 是 政治 斗争 形式 之后 先后 出现 反对 新政府 返还 桑解阵 政府 没收 土地 大规模 罢工 对 国内 和解 政策 不满 中部 地区 骚乱 以及 前桑 解阵 一批 成员 于 年 月 在 新塞哥 维亚 省 成立 名为 丹托 消灭 复仇主义 游击队 新政府 力图 以 谈判 等 和解 方式 解决 这些 问题 年月日 在 尼加拉瓜 首都 马那瓜 以北 马塔 加尔帕市 举行 反 政府 武装力量 支出 武器 仪式 其中 既有 新政府 成立 后 已 解除武装 后来 因 政府 未能 履行 安置 协议 又 重新 拿 起 武器 人 又 有 反对派 桑解阵 中 因 对 新政府 不满 而 武装 反抗 人 自此 尼加拉瓜 解除 反 政府 武装 工作 取得 阶段性 进展 在 此后 年 里 小规模 武装冲突 仍 接连不断 如 美籍 古巴人 托尼 布赖恩 特 领导 行动 小组 反 政府 武装 组织 北方 阵线 亲桑 解阵 武装 组织 主权 和 尊严 别动队 工农 革命 阵线 等 不断 绑架 和 暗杀 活动 甚至 对 政府 武装 攻击 直到 年月日 尼加拉瓜 总统 阿莱曼 才 与 北方 阵线 签订 和平 协议 反 政府 武装 交出 武器 重返 社会 与 尼加拉瓜 相比 萨尔瓦多 和平 进程 进展 较为 顺利 自从 年月日 政府 与 法拉本多 马蒂 民族 解放阵线 领导 游击队 签署 和平 协议 以来 虽然 时有 摩擦 但 双方 基本上 能 恪守 和平 协议 成立 萨尔瓦多 巩固 和平 委员会 在 联合国 监督 下 游击队 分批 完成 全部 遣散 政府 也 解散 专门 用来 对付 游击队 个 快速反应 营月 日 政府 和 法拉本多 马蒂 民族 解放阵线 举行 庆祝 仪式 萨尔瓦多 内战 宣告 结束 在此之前 月 日 萨尔瓦多 总统 克里斯蒂亚尼 和 洪都拉斯 总统 卡列 哈斯 发表 联合公报 重申 尊重 和 履行 国际法庭 对 两国 边境 有 争议 领土 裁决 从而 初步解决 多年 萨尔瓦多 洪都拉斯 领土 之争 在 以后 几年 里 除少数 几起 暗杀 活动 外 萨尔瓦多 并 没有 像 尼加拉瓜 那样 发生 武装冲突 政局 平稳 在 危地马拉 直到 年月日 全国 革命 联盟 司令 们 同 代表 政府 和平 委员会 才 达成 最后 和平 协议 危地马拉 和平 进程 进展 也 相对 顺利 年月日 危地马拉 最后 一批 前 游击队员 被 遣散 从而 结束 长达 年 内战 第二 中美洲 民主化 进程 与 和平 进程 是 相辅相成 年 中美洲 国 执政党 在 总统 选举 中 接连 失利 反对党 纷纷 上台 年 萨尔瓦多 右翼 政党 民族主义 共和 联盟 在 大选 中 胜利 大大降低 右翼 势力 对 强权 人物 心理 诉求 增强 工商 阶层 对 和平 与 民主 信心 注民欣 一平 中美洲 告别 战乱 载 世界 知识 年 第期 和平 协议 签署 以后 法拉本多 马蒂 民族 解放阵线 向 萨尔瓦多 最高 选举 法庭 递交 注册 申请报告 宣布 正式 成立 法拉本多 马蒂 民族 解放阵线 党 参加 年 举行 总统 和 议会选举 年 月 萨尔瓦多 多万 选民 参加 内战 结束 后 第一次 全国 大选 执政 民族主义 共和 联盟 得票 最多 其 候选人 阿曼 多 卡尔德隆 索尔 当选 为 总统 法拉本多 马蒂 民族 解放阵线 党 赢得 席 超过 传统 第二 大 党 基督教 民主党 成为 萨尔瓦多 第二 大 政治 力量 在 年 地方 选举 中 民族民主 共和 联盟 得票数 位居 第一 成为 萨尔瓦多 第一 大 党 民族主义 共和 联盟 则 退居 第二位 在 尼加拉瓜 虽然 桑解阵 在 年 大选 中 失利 但 仍 是 仅次于 全国 反对派 联盟 重要 政治 力量 在 大选 失利 后桑 解阵 宣布 做 建设性 反对派 表示 支持 新政府 一切 有利于 人民 政策 桑解阵 许多 领导人 在 政府 和 军队 中 任职 但 在 年 月 尼加拉瓜 政府 因 美国 压力 迫使 温贝托 奥尔特加 退役 全国 反对派 联盟 分裂 桑解阵 成为 尼加拉瓜 第一 大 政党 占据 议会 以上 席位 在 年 月 地方 选举 中桑 解阵 又 取得胜利 为 其 在 年 大选 中 争取 获胜 重返 政坛 创造 在 危地马拉政府 和 游击队 组织 全国 革命 联盟 在 签署 和平 协议 之前 就 达成 关于 修改 宪法 和 选举法 协议 规定 全国 革命 联盟 将 合法 政党 参加 国内 政治 生活 年代 以来 中美洲 国家 兴起 修宪 运动 达到 巩固 和平 发展 经济 打击犯罪 取消 军队 更改 官员 任期 和 更新 国家 观念 等 目标 中美洲 各国 纷纷 修改 宪法 年月日 尼加拉瓜 总统 查莫罗 夫人 签署 法令 颁布 新 宪法 修改 年 宪法 第款 中 第条 新 宪法 规定 禁止 总统 亲属 参政 及 参加 下届 总统 竞选 总统 不能 连任 任期 年 改为 年 总统 竞选 如 没有 名 候选人 达到 有效票 须 第轮 选举 取消 义务兵役制 改组 桑地诺 人民军 为 国民军 建立 人权 检查 机构 总统 在 做出 重大 决定 前 必须 同 议会 协商 等等 在 危地马拉 埃利亚 斯 继任者 拉米罗 德 莱昂 在 年 月 发起 修宪 运动 把 总统 和 议员 任期 从 年 改为 年 法官 任期 从 年 改为 年 决定 由 中央银行 加强 对 公共开支 控制 萨尔瓦多 于 年 通过 宪法 修正案 使 司法 有 更 多 自主权 巴拿马 则 于 年 通过 取消 军队 决定 规定 巴拿马运河 回归 之后 管理 事项 哥斯达黎加 也 修改 宪法 包括 打击 破坏 生态环境 行为 和 允许 警方 干预 涉嫌 毒品走私 和 洗钱 电话 此外 哥斯达黎加 对 允许 总统 连任 提案 讨论 这些 修宪 运动 为 中美洲 民主化 发展 创造 有利条件 值得注意 是 在 内战 结束 以后 中美洲 各国 政治 进程 中 中美洲 各国 尽管 受到 国内 不同 政治派别 斗争 干扰 但 这些 国家 法制 不断完善 在 程度 上 确保 民主化 进程 发展 说 资产阶级 民主 制度 在 中美洲 国家 已经 确立 尽管 处在 实习期 但 从 长远 来看 制度 逐渐 稳定 发展 是 不容置疑 第三 伴随 民主化 进程 发展 中美洲 各国 传统 政治 力量 出现 分化 改组 但 一些 国家 政坛 仍 由 两大 政党 唱主角 大选 后 不久 尼加拉瓜 桑解阵 中 一些 不满 该 阵线 政策 人 建立 丹托 消灭 复仇主义 游击队 北方 阵线 等 组织 在 程度 上 消弱 桑解阵 年 月 占 议会 以上 议席 成为 尼加拉瓜 第一 大 政党 桑解阵 在 议会 中 分裂 成 两个 党团 年 桑解阵 部分 领导人 退出 桑解阵 组建 桑地诺 革新运动 另 立新 党 又 一次 削弱 桑解阵 年 月 尼加拉瓜 全国 反对派 联盟 宣布 开除 联盟 成员 基督教民主联盟 保守 人民 联盟 和 民主运动 党 执政党 发生 分裂 尽管如此 尼加拉瓜 国内 两极 政治 并未 发生 根本 改变 在 萨尔瓦多 执政 民族主义 共和 联盟 与 法拉本多 马蒂 民族 解放阵线 和 基督教 民主党 实力 本来 就 不是 很 悬殊 年 月 虽然 萨尔瓦多 人民 革新 论坛 退出 法拉本多 马蒂 民族 解放阵线 但 萨尔瓦多 并未 出现 多元化 政治 格局 类似 情况 还有 危地马拉 政党 制度 比较 健全 哥斯达黎加 自 世纪 年代 中期 以来 出现 基督教 社会 团结党 和 民族 解放党 轮流 执政 情况 中美洲 其他 国家 如 巴拿马 和 洪都拉斯 左派 力量 比较 分散 失去 与 政治 对手 分庭抗礼 能力 但 在 尼加拉瓜 危地马拉 和 萨尔瓦多 国内 政局 均 由 阵线 较为 分明 两派 力量 所 控制 既 没有 出现 与 西方 国家 类似 第三条 道路 也 没有 建立 中间派 联盟 第四 中 左派 政党 在 中美洲 式微 但 仍 是 不可 忽视 重要 政治 力量 同时 也 出现 一些 值得注意 变化 年 中美洲 国 执政党 在 总统 选举 中 接连 失利 右派 反对党 纷纷 上台 它们 在政治上 亲美 反共 在经济上 推进 私有化 而 左派 在 保卫 过去 斗争 成果 同时 并 没有 拿出 发展 社会 经济 济世 良方 如 尼加拉瓜 桑解阵 虽然 激烈 反对 政府 私有化 经济 改革 但 也 没有 拿出 令人信服 解决 当前 严重 社会 经济 问题 方案 哥斯达黎加 驻 中美洲 国家 大使 路易斯 索利斯 左派 已经 没有 能力 使 计划 焕然一新 本 地区 几个 国家 中 出现 中 左派 政府 已经 明显 地 丧失 力量 而且 被 新 自由主义 所 摧毁 注何 塞梅 嫩 德 路易斯 索利斯 说 中美洲 左派 政府 明显 丧失 力量 载 墨西哥 至 上报 年月日 虽然 中美洲 保护 人权委员会 总 协调员 丹尼尔 卡马乔 下述 说法 有失 偏颇 但 也 并非 毫无道理 他 说 各种 社会 政治 进程 和 民主化 趋向 扫除 那些 失去 威信 不 合法 没有 能力 控制 社会 和 缺少 执政 能力 政府 注何 塞梅 嫩 德 路易斯 索利斯 说 中美洲 左派 政府 明显 丧失 力量 载 墨西哥 至 上报 年月日 中美洲 左派 中间 确实 着 学说 含糊不清 和 思想 危机 注何 塞梅 嫩 德 路易斯 索利斯 说 中美洲 左派 政府 明显 丧失 力量 载 墨西哥 至 上报 年月日 在 经济 问题 成为 社会 发展 瓶颈 阶段 除非 右派 政府 政策 出现 大 失误 或 国际形势 发生 重大 变化 选民 很难 选择 被 缺乏 执政 能力 政党 但 值得注意 是 到 世纪 之初 左派 力量 得到 恢复 与 发展 尤其 是 尼加拉瓜 桑解阵 和 萨尔瓦多 法拉本多 马蒂 民族 解放阵线 在 国内 都 成为 第一 大 党 那种 到 年内 拉美 多数 国家 将 由 左派 执政 预见 并非 完全 没有 可能 注 徐世澄 拉美 政党 新 趋向 载 当代世界 年 第期 第页 三 中美洲 政治形势 未来 发展趋势 笔者 未来 中美洲 政治形势 将 出现 如下 发展趋势 第一 中美洲 政局 继续 趋向 稳定 武装斗争 不再 是 政治 斗争 形式 世纪 年代 战乱 给 中美洲 留下 伤痕 是 难以 用 数字 来 计算 萨尔瓦多 洪都拉斯 和 危地马拉 等国 游击队 在 坚持 十几年 甚至 几十年 武装斗争 之后 同 政府 展开 谈判 随后 放下 武器 其 原因 在于 这些 国家 人民 不 支持 战乱 各派 力量 也 认识 到 深受 武装冲突 之害 中美洲 只有 加入 寻求 繁荣 民主 与 和平 民主 世界 才 成为 这场 不幸 和 错误 战争 牺牲者 所 接受 惟一 供品 也许 是 表明 死 并非 毫无用处 惟一 方式 注 曼努埃尔 德 莱昂 再见 中美洲 武器 载 美国 世界 新闻 报 年月日 世纪 初 已 在 国内 民主 制度 下 活动 政治 反对派 是 前 游击队 没能 提出 令人信服 解决 复杂 社会 经济 问题 方案 政权 仍 牢牢地 掌握 在 右派 手中 前 游击队 政治势力 上台 可能性 不 大 即使 是 左派 上台 也 不会 冲破 既有 资本主义 民主 架构 此外 从 国际 环境 来看 后 冷战 时代 市场经济 全球化 浪潮 波及 世界 每个 角落 各国 政府 在 国内 政治 稳定 前提 下 应对 和 参与 吸引 国外 投资 从而 解决 国内 问题 中美洲 各国 都 由 右派 当权 美国 等 西方 国家 希望 中美洲 政局稳定 以 减少 对 它们 无偿援助 刺激 直接 投资 取得 双赢 总之 以 武装斗争 解决 分歧 可行性 愈来愈 小 从 中美洲 区域 政治 形势 来看 确实 不少 问题 尼加拉瓜 和 洪都拉斯 领土 争端 再起 尼加拉瓜 与 哥斯达黎加 关于 圣胡安河 航行 争端 危地马拉 向 伯利兹 提出 领土 尼加拉瓜 萨尔瓦多 和 洪都拉斯 关于 丰塞卡 湾 划界 问题 尚未 解决 巴拿马 境内 哥伦比亚 游击队 带来 不安 等等 所有 这些 问题 都 会 对 地区 稳定 消极影响 但 从 中美洲 面临 任务 社会 稳定 与 经济 发展 国际 环境 国际 关系 改善 和 一体化 以及 各国 采取 解决问题 方式 来看 这些 问题 不会 影响 中美洲地区 宏观 稳定 如 洪都拉斯 尼加拉瓜 两国 已 将 分歧 提交 国际法庭 仲裁 洪都拉斯 用 美洲国家组织 和平 基金 资助 成员国 解决 冲突 哥伦比亚 也 将 其 与 洪都拉斯 等国 关税 纠纷 提交 世贸组织 等等 注 第二 中美洲 资产阶级 民主 制度 将会 稳定 发展 为 国内 各 政治 力量 包括 左派 提供 政治 活动 基本 框架 实际上 到 年 中美洲 各国 都 有 选举 日程表 民众 参与 水平 在 年代 以来 有所提高 如 哥斯达黎加 民众 参与率 超过 危地马拉 年代 以来 民众 参与率 在 左右 注 总的来看 尽管 中美洲 民主化 程度 很 低 定期 选举 并 不 意味着 民主 和 法制 加强 但 中美洲 人民 毕竟 看到 反对派 因 选举 而 上台 不合时宜 政府 因 选举 而 下台 情景 这 本身 就是 进步 注年 中美洲 国 伯利兹 除外 共 举行 次 大选 形式 上 民主化 和 选举 进程 虽然 不足以 摆脱 危地马拉 萨尔瓦多 洪都拉斯 和 尼加拉瓜 等国 严重 政治 和 经济危机 但 至少 进程 使 左派 有 明确 合法性 和 法制性 趋向 政府 也 得到 社会 普遍 认可 解决 政府 和 反对派 合法性 危机 就 为 国家 克服 经济危机 创造 国家 民主 制度 为 各种 政治 力量 所 提供 反映 自己 意见 舞台 有利于 国内 政治 稳定 反过来 稳定 又 会 促进 原有 制度 框架 完善 与 发展 中美洲 近年来 经济 发展 虽然 没有 根本 解决 各国 普遍存在 贫困 问题 但 也 确实 为 政治 稳定 和 民主化 进程 奠定 基础 民众 民主 参与 意识 逐渐 增强 在 各国 修宪 过程 中 各派 政治 力量 争夺 激烈 而且 在 选举 中 普通 民众 政治 参与 热情 也 有 提高 从 国际 方面 来看 前 苏联 这样 地缘 政治 争夺者 已 不复存在 对 中美洲 有 重大 影响 美国 等 西方 国家 对 中美洲 直接 干预 减少 它们 从 其 全球战略 出发 转而 注意 促进 中美洲 民主化 此外 联合国 有关 机构 影响 尤其 是 中美洲 一体化 谈判 逐渐 深入 对 国内 民主化 进程 将 起到 积极 作用 中美洲 议会 等 一体化 组织 和 国际 条约 会 直接 影响 民众 利益 促进 在 中美洲 政治 中 民族 认同 和 意识形态 建议 潜力 注 第三 毒品 移民 和 社会 暴力 是 困扰 中美洲 国家 大 社会 问题 新 自由主义 改革 带来 贫困 失业 等 严重 经济 问题 拉美国家 当务之急 是 解决 贫困 和 失业问题 毒品走私 泛滥 危害 人民 健康 直接 影响 拉美国家 与 美国 等 西方 国家 政治 关系 中美洲 非法 移民 大量 涌入 美国 给 中美洲 国家 与 美国 外交关系 带来 阴影 而且 直接 影响 美国 援助 与 投资 影响 中美洲 国家 经济 日益 增多 暗杀 绑架 和 其他 社会 暴力 威胁 人民 生命财产 安全 而且 冲击 国家 政治 稳定 据 欧盟 和 拉美国家 民意调查 机构 调查 显示 年 有 中美洲 人 民主 政府 不够 完善 并且 有 人 赞成 独裁政权 注 刘新民 拉美 在 稳定 中 继续 深化 政治 改革 载 拉丁美洲 研究 年 第期 第四 在 国际 关系 方面 中美洲 国家 外交 方向 重点 放在 拉美 尤其 是 中美洲地区 国家 以 加速 地区 一体化 进程 解决 共同 面临 社会 经济 问题 中美洲 首脑会议 频繁 召开 和 解决问题 多样性 是 一体化 谈判 重要 特点 欧盟 对 中美洲 表现 出 浓厚 兴趣 在 今后 双边关系 中 经贸关系 将 更加 密切 加入 北美自由贸易区 是 中美洲 各国 外交 长远目标 墨西哥 特殊 地理位置 和 与 中美洲 相似 文化背景 使 它 成为 中美洲 国家 最 重要 外交 对象 之一 墨西哥 对 中美洲 颇为 重视 自年 以来 塞迪略 政府 就 与 中美洲 签订 个 议定书 承诺 向 中美洲地区 提供 亿美元 技术 援助 年 月 尚未 就任 墨西哥 当选 总统 福克斯 就 提出 未来 向南看 与 中美洲 国家 一起 为 共同 目标 而 工作 注')
labels = torch.tensor([8, 8, 9, 6])

# /home/oem/mydisk/outer_models/cn_models/chinese_xlnet_mid_pytorch

new_corpus = []
for single_doc in corpus:
    single_doc = single_doc.replace(' ', '')
    new_corpus.append(single_doc)
    print(len(single_doc))

inputs = tokenizer(new_corpus, return_tensors='pt', padding=True, truncation=True, max_length=512)
labels = labels.unsqueeze(0)

outputs = model(**inputs, labels=labels)
loss = outputs.loss
logits = outputs.logits

print(f"loss: {loss} logits: {logits}")
print(logits.shape)
'''

cn_bert_wwm_model_path = '/home/oem/mydisk/outer_models/cn_models/chinese-bert_chinese_wwm_pytorch'
cn_xlnet_path = '/home/oem/mydisk/outer_models/cn_models/chinese_xlnet_mid_pytorch'

# 使用bert_wwwm模型
# cn_bert_tokenizer = BertTokenizer.from_pretrained(cn_bert_wwm_model_path)
# cn_bert_model = BertForSequenceClassification.from_pretrained(cn_bert_wwm_model_path)

# 使用xlnet模型
cn_xlneet_tokenizer = XLNetTokenizer.from_pretrained(cn_xlnet_path)
cn_xlnet_clsfy_model = XLNetForSequenceClassification(cn_xlnet_path)


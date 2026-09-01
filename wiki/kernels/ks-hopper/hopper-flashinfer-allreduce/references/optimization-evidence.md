# FlashInfer AllReduce 实现与效果基线

## 可复用实现模式

不要按历史文件路径寻找修改点。先在当前推理框架中沿 AllReduce 调用链识别四类职责，再把优化规则映射到对应职责：

1. **容量计算**：MiB、workspace 与 max token 计算保持整数，兼容小数 MiB 阈值。
2. **配置传播**：fused 与 standalone 路径读取同一 payload 上限；配置必须在通信器或执行图初始化前生效。
3. **资格判断**：仅让 CUDA、contiguous、2-D、受支持 dtype 且阈值内的张量进入专用路径。
4. **路由与回退**：显式启用的 eligible FlashInfer 优先于通用节点内路径；其他输入继续使用原有 backend。

在新代码库中，可搜索 AllReduce 入口、backend 选择、workspace 初始化、payload 阈值与 fallback 分支来定位这些职责，但不要假设固定目录、类名或配置字段。

## 常见失效模式

- float token 数进入 FlashInfer workspace API 后可能触发 `TypeError: unsupported operand type(s) for &: 'float' and 'int'`。
- fused 与 standalone 路径可能使用不同阈值或不同配置生命周期。
- 即使用户显式选择 FlashInfer，符合条件的张量也可能被更早的通用路由消费。

## 已观测效果

实现验证应覆盖小数阈值整数化、payload 边界、支持与不支持输入、专用路由优先级及所有 fallback。以下数据仅用于说明该模式曾产生的效果量级，不作为新环境的验收承诺。

单机 8×H200、TP8、hidden size 7168、BF16，warmup 100、measure 500 的单算子中位数：

| tokens | SymmMem | FI MNNVL | 延迟下降 |
|---:|---:|---:|---:|
| 36 | 10.733 us | 5.728 us | 46.63% |
| 64 | 13.418 us | 7.754 us | 42.21% |
| 112 | 16.230 us | 11.405 us | 29.73% |
| 128 | 17.078 us | 12.352 us | 27.67% |

真实模型 trace：baseline 的 MNNVL kernel 为 0 次、multimem 为 615 次；candidate 的 MNNVL kernel 为 35 次、multimem 仍为 615 次。该结果证明路由生效，也证明 2 MiB 仅覆盖部分小张量。

端到端首次正式 paired 结果为 mixed：1K→1K 为 -1.560%，8K→1K 为 +1.064%。同机 1K→1K 十轮确认得到 candidate mean +4.008%、median +3.219%、paired median +2.534%，7/10 对为正。另一台 H200 的十轮复测中位数为 +4.223%，但存在异常 baseline 慢轮；敏感性分析后的 8 对 mean +2.749%、paired median +2.863%，6/8 为正。

## 结论边界

证据支持：整数缺陷已修复；显式阈值被 standalone 读取；符合条件的小张量优先进入 MNNVL；fallback 有效；目标 H200 上阈值内单算子有优势；真实模型路由生效。

证据不支持：固定 4% 端到端收益；把 2 MiB 设为所有用户默认值；所有 AllReduce 都切换到 FlashInfer；把 27%–47% 单算子延迟下降等同于模型吞吐提升。

因此能力定位为 `ACCEPT_EXPLICIT_OPT_IN`。部署前必须在目标机器、目标 workload 上重新做交替顺序的 paired A/B。

## 经验复用原则

- 把历史实现和实验记录视为知识来源，不把具体代码布局或实验编号写入 Skill 的触发条件。
- 复用实现不变量、验证流程和结论边界；具体性能数字只作为已观测效果，不作为验收承诺。
- 换 GPU 拓扑、模型、shape、并发、版本或阈值后，重新建立 baseline 并做交替顺序 paired A/B。

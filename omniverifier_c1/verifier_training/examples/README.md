# OmniVerifier C1 verifier training

先在独立环境中安装 `verifier_training`，设置训练集后运行：

```bash
export TRAIN_FILES=/absolute/path/to/pointwise_train.jsonl
bash examples/pointwise_c1.sh
# 单独运行 pairwise 实验：
export TRAIN_FILES=/absolute/path/to/pairwise_train.jsonl
bash examples/pairwise_c1.sh
```

`MODEL_PATH` 默认是 `Qwen/Qwen3-VL-8B-Instruct`，可用环境变量覆盖；`VERIFIER_PYTHON` 可指定解释器。
两个脚本都使用八卡 DAPO 配置，batch size 512、rollout n=8，最大输入 4096 token、最大输出 16384 token。
仅加载训练集并进行参数更新，每 5 个 optimizer step 保存 checkpoint；末尾不足保存间隔时补存。
两个实验使用不同名称保存日志和 checkpoint，不加载验证集，也不执行训练前、中、后的独立评测。

## 数据字段

每行包含 `problem`（完整规则和检查要求）、`task`、`answer`（GT）、`initial_image`（原图）和 `images`（一张生成图）。
Pairwise 另外包含一张 `reference_image`（正确终态）。每个图片字段只允许一张图，可以是单张路径或只有一个元素的列表。
相对图片路径通过 `data.image_dir` 解析；也支持现有加载器可处理的图片字节/PIL 对象。

Pointwise 示例：

```jsonl
{"problem": "Move the player one cell right. Preserve all other cells.", "initial_image": "initial.png", "images": ["generated.png"], "task": "global_score", "answer": "0.5"}
{"problem": "Is the player in the cell one step right of its original position?", "initial_image": "initial.png", "images": ["generated.png"], "task": "fine_grained_verification", "answer": "True"}
```

Pairwise 示例：

```jsonl
{"problem": "Move the player one cell right. Preserve all other cells.", "initial_image": "initial.png", "images": ["generated.png"], "reference_image": "reference.png", "task": "global_score", "answer": "0.5"}
```

混合两种 task 时，示例统一使用字符串 `answer`，避免列类型冲突；数值 GT 和包含 `global_score` 的 JSON GT 也可使用。

输入自动插入图片占位，顺序与生成模型奖励服务一致：pointwise 是 `Initial Image`、`Generated Image`；pairwise 再追加 `Reference Image`。
`problem` 不需要 `<image>`。原图和生成图是必填字段，pairwise 还必须有参考图。
这套处理同时用于训练和 prompt 长度过滤。共享 Jinja 模板会根据 mode 选择原图推导终态或参考图终态，根据 task 选择全局评分或二值输出。

## 输出与奖励

两个阶段的全局分数字段统一为 `global_score`。全局输出只包含 JSON：

```json
{
  "sub_questions": [
    {"id": 1, "question": "Is the player in the required cell?", "answer": "True", "explanation": ""},
    {"id": 2, "question": "Are all other cells unchanged?", "answer": "False", "explanation": "One unrelated cell changed."}
  ],
  "global_score": 0.5
}
```

`global_score` 是满足原子规则的比例，保留两位小数，必须为 `[0, 1]` 范围内的 JSON 数值。
细粒度任务仍输出 `{"answer": "True"}` 或 `{"answer": "False"}`。
两种输出均不使用 thinking 标签、Markdown 代码围栏或 JSON 之外的文字。

- `global_score` 奖励：`0.1 × JSON格式奖励 + 0.9 × (1 - abs(预测分数 - GT))`。
- `fine_grained_verification` 奖励：`0.1 × JSON格式奖励 + 0.9 × 二值正确性奖励`。

JSON 格式奖励要求独立 JSON 对象内能提取有效的 task 对应预测。导出方法见 [verifier README](../README.md)。

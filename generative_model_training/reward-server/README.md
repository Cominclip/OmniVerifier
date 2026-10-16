# OmniVerifier reward server

两种实验均使用八卡训练 Qwen-Image-Edit-2509，另外八卡部署对应 verifier，算法为 DiffusionNFT，rollout n=8。Pointwise 与 pairwise 分别运行，每次部署 8 个单 GPU vLLM 副本，`tensor_parallel_size=1`。

下文命令均从仓库的 `generative_model_training/` 目录执行：

```bash
cd generative_model_training
```

## 两台八卡机器

训练机 A 运行图像生成与 DiffusionNFT 参数更新；奖励机 B 运行 verifier。训练端通过 HTTP 发送 initial_image、生成图像、编辑指令，以及 pairwise 模式的 reference_image，奖励端返回同顺序的全局分数。

在奖励机 B 的 `generative_model_training/` 目录运行：

```bash
pip install -r reward-server/requirements.txt
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python reward-server/app_pointwise.py --host 0.0.0.0 --port 18085
# Pairwise 实验使用另一入口：
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python reward-server/app_pairwise.py --host 0.0.0.0 --port 18086
```

启动前填写 `reward-server/omniverifier_reward/config.py` 中对应的 verifier 模型路径，也可以使用 `POINTWISE_VERIFIER_MODEL_PATH` / `PAIRWISE_VERIFIER_MODEL_PATH` 环境变量。`reward-server/omniverifier_reward/prompts.py` 已分别内置 pointwise 和 pairwise 的游戏图像编辑评估 prompt，无需再填写空模板。

在训练机 A 的 `generative_model_training/` 目录运行，将 `REWARD_MACHINE_IP` 替换为奖励机的可达 IP：

```bash
pip install -e .
export DATASET_DIR=/absolute/path/to/dataset
export START_REWARD_SERVER=0
export TRAIN_GPUS=0,1,2,3,4,5,6,7
export REWARD_SERVER_URL=http://REWARD_MACHINE_IP:18085
bash scripts/train.sh pointwise
# Pairwise：
export REWARD_SERVER_URL=http://REWARD_MACHINE_IP:18086
bash scripts/train.sh pairwise
```

`START_REWARD_SERVER=0` 表示连接已部署的奖励服务。两台机器只需对应 HTTP 端口可达；生成模型的八个训练进程仍在训练机内部同步，不需要把奖励机加入训练进程组或同一个 Ray 集群。图像以 PNG/base64 传输，两台机器不需要共享图片文件系统。

## 单台十六卡机器

```bash
export DATASET_DIR=/absolute/path/to/dataset
bash scripts/train.sh pointwise
# 或：
bash scripts/train.sh pairwise
```

默认 `START_REWARD_SERVER=1`：GPU `0-7` 训练，GPU `8-15` 部署奖励服务，通过本机地址通信。可以设置 `TRAIN_GPUS` 和 `REWARD_GPUS` 修改编号，本机部署时两个列表需要各含 8 张不同且互不重叠的卡。脚本等待奖励服务就绪后开始训练，并在退出时停止自己启动的服务。

每卡采样和训练 batch 均为 16。八卡每批生成 128 张图像，包含 16 条输入各 8 次 rollout；每收集两批候选图后执行一次 optimizer 更新，完整批次的有效训练批量为 256。

默认训练一个完整 epoch：打乱训练数据后，每条输入访问一次，每条生成 8 张候选图。若有 N 条训练数据，且评分均有效，optimizer 更新次数为 `ceil(N / 32)`。末批不足时用带标记的占位样本对齐各 GPU 的 batch，占位样本不参与奖励统计和损失；最后不足两批的候选图也会执行更新。

生成器条件序列上限为 1024 token，包含图像和文字条件；token 长度与图片分辨率是独立参数，当前图片分辨率仍为 512×512。

仅执行训练，每 10 个 optimizer step 保存 checkpoint，训练结束时补存最后一次更新。已移除评测入口及其配置，不加载验证集或测试集。

## 数据与评分

`DATASET_DIR` 中需要非空的 `train_metadata.jsonl`。图片路径可以相对于数据目录，或使用绝对路径。仅需要训练集。

Pointwise 每行格式：

```json
{"prompt": "将杯子的颜色改成红色", "initial_image": "images/initial.png"}
```

Pairwise 每行格式：

```json
{"prompt": "将杯子的颜色改成红色", "initial_image": "images/initial.png", "reference_image": "references/target.png"}
```

`prompt` 字段应包含完整的游戏规则、编辑要求和约束。训练端将它作为 `instruction` 发送给奖励服务，并替换评估模板中的 `${game_rules}`，不会对模板中的 JSON 花括号做格式化。

Pointwise 按 Initial Image、Generated Image 的顺序传图，根据原图和规则推导预期终态。Pairwise 按 Initial Image、Generated Image、Reference Image 的顺序传图，以参考图为正确标准。Reference image 只参与奖励评测，不作为生成器输入。

两个模板均要求将规则独立分解成可视化检查的原子规则，仅输出包含 `sub_questions` 和 `global_score` 的 JSON，不使用 thinking 标签或 JSON 之外的文字。每条规则包含 `id`、`question`、字符串 `answer`（`"True"` / `"False"`）和 `explanation`（满足时为空字符串）。`global_score` 为满足规则的比例，保留两位小数，例如：

```json
{
  "sub_questions": [
    {"id": 1, "question": "Is the player in the expected cell?", "answer": "True", "explanation": ""},
    {"id": 2, "question": "Are all other cells unchanged?", "answer": "False", "explanation": "An unrelated cell changed."}
  ],
  "global_score": 0.5
}
```

奖励只读取顶层 `global_score`，要求它是 `[0, 1]` 范围内的有限 JSON 数值。Verifier 训练侧也使用相同的字段、独立 JSON 输出和图片顺序，不再接受其他旧版全局分数字段。

默认 verifier 输入 4096、输出 16384 token，总上下文 20480。请求按样本尽量平均分配给八个 worker，小批次轮转，返回结果恢复原顺序。`GET /health` 用于就绪与模式检查，`POST /score` 返回 `scores`、`valid` 和 `errors`。

输入超长或评分解析失败时返回 `score=null, valid=false`。这些结果从组均分、标准差、日志奖励均值和全部训练损失中排除；正常的 0 分仍有效。若一轮候选图的评分全部失败，所有训练卡共同跳过 optimizer 更新。各卡有效样本数不同时，损失按全局有效样本数归一化。

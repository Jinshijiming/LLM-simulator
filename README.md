# LLM 实验台

浏览器里的微型 GPT：用对话 JSON 做语料，现场训练、导出，再拿到生成页推理。没有后端，权重都在本地。

默认语料是电磁信号：调制、体制、辐射源个体识别。

## 运行

需要 Node.js 20+。不要直接双击 `index.html`，ES 模块会加载失败。

```bash
npm start
```

浏览器打开 [http://127.0.0.1:5173](http://127.0.0.1:5173)

自检：

```bash
npm test
```

## 页面

1. **语料**  
   标准 chat JSON。右侧是对话，下面是打包序列（`sys` / `usr` / `ast` / `end`）和词表。
2. **结构**  
   示意图：嵌入、带 Q/K/V 与因果注意力的层、残差、解嵌入。点模块看张量形状。可改层数、宽度、头数、上下文。
3. **训练**  
   选语料，Adam / SGD，看损失、batch、注意力、词嵌入。只在 assistant 段上算损失。
4. **生成**  
   选模型（当前训练或检查点），温度 / Top-k / Top-p，逐步或连续采样。

## 训练到生成

1. 训练页选语料（电磁 / 调制 / 体制 / 个体，或语料页自定义 JSON）。
2. 训练。
3. **导出模型**，得到 JSON，同时出现在生成页列表。
4. 生成页选该检查点，或 **导入** 之前的 JSON。
5. 输入提示再生成。

选中检查点后，生成用的是当时冻住的权重，不会被继续训练改掉。「当前训练」则跟正在训的模型是同一份。

## 语料格式

```json
[
  {
    "messages": [
      { "role": "system", "content": "你是电磁信号分析助手。" },
      { "role": "user", "content": "BPSK是什么调制？" },
      { "role": "assistant", "content": "二进制相移键控，两相位相差180度。" }
    ]
  }
]
```

也接受单条对象，或扁平的 `{ "system", "user", "assistant" }`。打包模板：

```
<|system|>…<|end|><|user|>…<|end|><|assistant|>…<|end|>
```

监督信号只加在 assistant 的标记、正文和结束符上。

预设：

| 名称 | 内容 |
| --- | --- |
| 电磁 | 调制 + 体制 + 个体 |
| 调制 | BPSK / QPSK / 16QAM / OFDM / MSK |
| 体制 | 脉压、FMCW、跳频、直扩、脉冲多普勒、相控阵 |
| 个体 | 射频指纹、功放非线性、杂散、相位噪声、脉内无意调制 |

这是教学用小语料。词表是「特殊标记 + 字符」，模型很小，适合看流程，不适合当真的电磁识别器。

## 模型

字符级 GPT-2 / Llama 混合风格，CPU 上现算：

- token 嵌入 + 位置嵌入
- RMSNorm
- 多头因果注意力
- SiLU FFN（4× 扩展）
- 解嵌入与 token 嵌入权重共享
- AdamW 或 SGD，梯度裁剪

默认约 2 层、宽 32、4 头、上下文 24。改结构会重建权重。

导出 JSON 含：`hparams`、`vocab`、语料快照、步数、损失、权重。magic 为 `llm-lab-ckpt-1`。

## 目录

```
index.html
server.js          本地静态服务，默认 5173
css/app.css
js/app.js          界面与训练/生成流程
js/gpt.js          前向、反向、Adam
js/tokenizer.js    特殊标记 + 字符
js/datasets.js     语料、打包、batch
js/sample.js       softmax / top-k / top-p
js/diagram.js      结构示意图
js/viz.js          损失、注意力、嵌入
js/selftest.js
```

## 注意

- 服务停了会连不上 `5173`，再执行 `npm start`。
- 语料太长或字符太杂，词表变大，训练会变慢、更难拟合。
- 检查点存在浏览器 `localStorage`（最多 8 个）以及你下载的文件里。

# WeiqiDetect

围棋棋盘拍照识别服务（Flask + OpenCV + **CNN**），部署于**微信云托管**。

## 识别流程

1. OpenCV 找棋盘四边并透视校正
2. 在每个交叉点裁剪 patch
3. **CNN（ONNX）** 三分类：空 / 黑 / 白
4. 返回棋子坐标矩阵

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查（含 `detector: cnn`） |
| POST | `/api/v1/detect` | 识别棋盘 |

### 请求

```json
{
  "imageUrl": "https://...",
  "boardSize": 19,
  "threshold": 0.52
}
```

### 响应

```json
{
  "ok": true,
  "detector": "cnn",
  "boardSize": 19,
  "black": [[3, 3]],
  "white": [[15, 15]],
  "confidence": 0.86,
  "stats": { "blackCount": 1, "whiteCount": 1, "modelValAccuracy": 0.99 }
}
```

## CNN 模型

- 构建时自动训练：Docker **多阶段构建** 在 `trainer` 阶段用合成棋面 patch 训练小型 CNN，导出 `models/stone_cls.onnx`
- 推理：`onnxruntime` CPU 推理，19×19 盘约 361 个 patch 批量预测
- 本地重训：

```bash
cd training
pip install -r requirements-train.txt
python train.py --output ../models/stone_cls.onnx --meta ../models/stone_cls.meta.json
```

后续可用真实棋谱照片做 fine-tune，替换 `models/stone_cls.onnx` 即可。

## 微信云托管部署

| 项 | 值 |
|---|---|
| AppID | `wx09786c3fb9eac9a4` |
| 环境 ID | `prod-d5g1g0dvr99bff1b5` |
| 服务名 | `weiqi-detect` |
| 端口 | `80` |

```powershell
wxcloud run:deploy --targetDir=. --dockerfile=Dockerfile --containerPort=80 --envId=prod-d5g1g0dvr99bff1b5 --serviceName=weiqi-detect --releaseType=FULL --noConfirm
```

**注意**：首次 CNN 版本构建会训练模型，耗时约 5–15 分钟，请耐心等待构建完成。

## 仓库

GitHub: `git@github.com:yoshikun/WeiqiDetect.git`

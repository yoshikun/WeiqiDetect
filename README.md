# WeiqiDetect

围棋棋盘拍照识别服务（Flask + OpenCV + Kaya Moku 辅助），部署于**微信云托管**。

## 识别流程（v6.1）

参考经典 OpenCV 围棋识别方案（[知乎专栏](https://zhuanlan.zhihu.com/p/347539186) 同类思路）与 [Kaya](https://github.com/kaya-go/kaya)：

### 默认 pipeline（`kaya`）

1. **主路径**：Moku RT-DETR（`moku-v3`）检测四角 + 棋子中心 → 透视映射到网格
2. **回退路径**：经典 CV（饱和度四角 + 交叉点 K-means 判子）
3. **手动校准**：用户拖动四角后，用经典 CV 在校正图上识子

### OpenCV 经典 pipeline（`opencv-series`）

小程序设置选 **OpenCV** 时启用，纯 OpenCV、无深度学习：

1. HSV 木色掩码 + 轮廓定位棋盘四角
2. 透视校正到正方形
3. 固定网格逐格采样，按黑白像素占比判子

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查（`version`, `pipeline: moku-rtdetr+cv`, `mokuReady`） |
| POST | `/api/v1/corners` | 自动建议四角坐标 |
| POST | `/api/v1/detect` | 识别棋盘（可选 `corners`, `withPreview`） |

### 识别请求

```json
{
  "imageUrl": "https://...",
  "boardSize": 19,
  "pipeline": "kaya",
  "corners": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]],
  "withPreview": true
}
```

### 健康检查响应

```json
{
  "ok": true,
  "version": "6.0.0",
  "detector": "kaya",
  "pipeline": "moku-rtdetr+cv",
  "mokuReady": true
}
```

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

构建时会从 HuggingFace 下载 `kaya-go/moku-v3` ONNX（约 77MB），首次构建约 5–15 分钟。

Moku 模型为 AGPL-3.0，商用请注意许可证。

## 仓库

GitHub: `git@github.com:yoshikun/WeiqiDetect.git`

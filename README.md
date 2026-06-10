# WeiqiDetect

围棋棋盘拍照识别服务（Flask + OpenCV + Kaya Moku 辅助），部署于**微信云托管**。

## 识别流程（v5.0 Baduk）

参考 Baduk Cap / Kaya 工作流：

1. 自动检测棋盘四角（饱和度分割，Moku 辅助）
2. 用户可在小程序校准页手动拖动四角
3. 透视校正到 800×800
4. 交叉点 K-means 判黑白子
5. 返回 9/13/19 路棋子坐标

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查（`version`, `detector: baduk`, `mokuReady`） |
| POST | `/api/v1/corners` | 自动建议四角坐标 |
| POST | `/api/v1/detect` | 识别棋盘（可选 `corners`, `withPreview`） |

### 识别请求

```json
{
  "imageUrl": "https://...",
  "boardSize": 19,
  "corners": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]],
  "withPreview": true
}
```

### 健康检查响应

```json
{
  "ok": true,
  "version": "5.0.0",
  "detector": "baduk",
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

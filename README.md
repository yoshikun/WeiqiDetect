# weiqi-detect

围棋棋盘拍照识别服务（Flask + OpenCV），部署于**微信云托管**，供小程序 `wx.cloud.callContainer` 调用。

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/api/v1/detect` | 识别棋盘 |

### 请求

```json
{
  "imageUrl": "https://...",
  "boardSize": 19
}
```

也支持 `image`（base64），但请求体大，不推荐。

### 响应

```json
{
  "ok": true,
  "boardSize": 19,
  "black": [[3, 3]],
  "white": [[15, 15]],
  "confidence": 0.5
}
```

## 微信云托管部署

| 项 | 当前项目值 |
|---|---|
| AppID | `wx09786c3fb9eac9a4` |
| 环境 ID | `cloud1-d4gjca9712bee12dc` |
| 服务名 | `django-nefm` |
| 端口 | `80` |

### 从 GitHub 拉取部署（推荐）

1. [cloud.weixin.qq.com](https://cloud.weixin.qq.com/) → 云托管 → 服务 `django-nefm`
2. **新建版本** → 代码来源选 **GitHub**
3. 授权并选择本仓库，分支 `main`，Dockerfile 路径 `Dockerfile`
4. 端口 **80**，全量发布

### CLI 部署

```powershell
npm i -g @wxcloud/cli
wxcloud login --appId wx09786c3fb9eac9a4 --privateKey "CLI私钥"
wxcloud run:deploy --targetDir=. --dockerfile=Dockerfile --containerPort=80 --envId=cloud1-d4gjca9712bee12dc --serviceName=django-nefm --releaseType=FULL --noConfirm
```

## 本地调试

```bash
pip install -r requirements.txt
python app.py
```

## 仓库

GitHub: `git@github.com:yoshikun/WeiqiDetect.git`

当前为 OpenCV 基础采样版（正拍、棋盘居中效果较好）。后续可增强透视校正、网格检测与置信度输出。

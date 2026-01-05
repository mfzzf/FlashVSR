# FlashVSR FastAPI 服务

## 启动

安装依赖（建议在 Python 3.10/3.11 环境）：

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
pip install -e . --no-deps
pip install -r api/requirements.txt
```

权重目录默认取：

- `examples/WanVSR/FlashVSR`（v1）
- `examples/WanVSR/FlashVSR-v1.1`（v1.1）

也可以通过环境变量覆盖：

- `FLASHVSR_V1_DIR=/models/FlashVSR`
- `FLASHVSR_V1_1_DIR=/models/FlashVSR-v1.1`

启动服务：

```bash
uvicorn flashvsr_api.app:app --host 0.0.0.0 --port 8000
```

## OpenAPI

- Swagger UI：`GET /docs`
- OpenAPI JSON：`GET /openapi.json`

## 推理接口

`POST /v1/upscale`（`multipart/form-data`）

- `file`: 输入视频（`.mp4/.mov/.avi/.mkv`）或帧压缩包（`.zip`，包含图片序列）
- `version`: `v1` 或 `v1.1`
- `pipeline`: `tiny` / `tiny_long` / `full`

示例：

```bash
curl -X POST "http://localhost:8000/v1/upscale" \
  -F "file=@./inputs/example0.mp4" \
  -F "version=v1.1" \
  -F "pipeline=tiny" \
  -o output.mp4
```

## k8s 探针

- 存活探针：`GET /healthz`
- 就绪探针：`GET /readyz`

# 🍅 小番茄图片解混淆 Telegram Bot

把 [MadLife77/xiaofanqie-image-confuse](https://github.com/MadLife77/xiaofanqie-image-confuse)
的希尔伯特曲线图片混淆算法移植到 Python，并封装成一个 Telegram Bot：
**收到混淆图片 → 解混淆 → 返回图片文件。**

## 算法说明

这个算法**不是加密**，只是一种像素置换（打乱），没有密钥。

1. **希尔伯特曲线**：`gilbert2d(width, height)` 生成一条广义希尔伯特曲线，
   把图片的每个像素恰好访问一次，得到像素位置的置换。支持任意长宽比（不必是正方形）和奇数尺寸。
2. **偏移量**：`offset = round((√5 − 1) / 2 × width × height)`，即约 0.618 × 总像素数
   （注意这里是 1/φ ≈ 0.618，不是常见的 φ ≈ 1.618）。
3. **置换**：

   ```
   加密: dst[curve[(i + offset) % N]] = src[curve[i]]
   解密: dst[curve[i]] = src[curve[(i + offset) % N]]
   ```

因为偏移量只由宽高决定，**任何知道图片尺寸的人都能还原**，不适合用于保密。

### 移植正确性

Python 实现与原 HTML 中的 JavaScript **逐像素完全一致**，由 `tests/test_algorithm.py`
中的差分测试保证：测试会把原 HTML 里的 `gilbert2d` 抽出来交给 Node 执行，
再和 Python 的结果逐点比对（覆盖 1×1 到 200×3 等 70 多种尺寸）。

> **移植陷阱**：JS 的 `Math.floor(ax / 2)` 对**负数**向下取整，Python 里必须用 `//`
> （同样向下取整），**不能**用 `int(ax / 2)`（向零截断）。非正方形图片的递归过程中
> 会出现负的 delta，用错就会在方图上"看起来正常"、在小图/长图上悄悄错乱。

## 安装与运行

### 方式一：本地 Python

```bash
cd xiaofanqie-bot
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

export TELEGRAM_BOT_TOKEN="123456:ABC..."   # 从 @BotFather 获取
.venv/bin/python bot.py
```

### 方式二：Docker（推荐）

```bash
# 构建
docker build -t xiaofanqie-bot .

# 运行（token 在运行时注入，不会被打进镜像层）
docker run -d --name xiaofanqie-bot \
  -e TELEGRAM_BOT_TOKEN="123456:ABC..." \
  --restart unless-stopped \
  xiaofanqie-bot
```

或用 compose（从 `.env` 读取 token）：

```bash
cp .env.example .env
# 编辑 .env 填入真实 token
docker compose up -d --build
docker compose logs -f
```

需要代理时（见下文「网络超时」）：

```bash
docker run -d -e TELEGRAM_BOT_TOKEN="..." \
  -e TELEGRAM_PROXY=http://host.docker.internal:7890 \
  --restart unless-stopped xiaofanqie-bot
```

镜像特点：

| 项目 | 说明 |
|---|---|
| 基础镜像 | `python:3.13-slim-bookworm`，与本项目开发测试的解释器一致 |
| 体积 | 约 318 MB（基础镜像 184 MB + numpy/Pillow 约 105 MB） |
| 运行用户 | 非 root（`botuser`, uid 10001） |
| 端口/卷 | 不需要。长轮询模式无需暴露端口，也无状态需持久化 |
| 健康检查 | `healthcheck.py` 会真的调一次 `getMe`，而不是只看进程是否存活 |
| 密钥 | `.env` 已在 `.dockerignore` 中排除，不会进入构建上下文或镜像层 |

> **健康检查为什么这么写**：本项目实际遇到的故障是「进程活着但每次请求都超时」，
> 所以只检查进程存活没有意义。`getMe` 能同时验证 token 有效和网络可达。
> 注意 Docker 不会因为 unhealthy 就自动重启容器（那需要 Swarm 或 autoheal），
> 它是给 `docker ps` 和监控看的信号。

> **挂载目录时的权限**：容器以 uid 10001 运行，如果要用 `-v` 挂载目录跑 `cli.py`，
> 宿主目录需要对该 uid 可写（例如 `chmod 777 <dir>`）。Bot 本身不需要挂载。

## 使用方法

**⚠️ 必须用「文件」方式发送图片，不能用「照片」。**

Telegram 会把「照片」自动压缩并缩放到长边约 1280px，这会破坏像素置换，
解混淆后只能得到乱码。正确做法：

1. 点击输入框的 📎 回形针
2. 选择「文件 / File」（不是「相册 / Gallery」）
3. 选中图片发送

如果误发照片，Bot 会提示改用文件方式。

### 命令

| 命令 | 说明 |
|---|---|
| `/start` / `/help` | 显示帮助 |
| `/id` | 显示当前 chat id |
| `/encrypt` | 切换加密/解密模式（默认解密） |

## 命令行使用（不依赖 Telegram）

```bash
# 解密（默认输出无损 PNG）
.venv/bin/python cli.py decrypt confused.jpg -o restored.png

# 加密
.venv/bin/python cli.py encrypt photo.png -o confused.png

# 批量处理到目录，输出 JPEG（有损，模仿原网页工具行为）
.venv/bin/python cli.py decrypt batch/*.jpg -d out/ --jpeg
```

## 网络超时 / 连接失败（重要）

如果遇到 `telegram.error.TimedOut` 或 `httpx.ConnectTimeout`（栈里通常是
`httpcore/_backends/anyio.py` 的 `start_tls`），**问题不在图片大小，而在连接建立阶段**。

### 实测根因

在编写本项目的机器上（到 `api.telegram.org` 的 **RTT 约 300ms**，TLS 握手 0.5～6 秒）：

| 现象 | 实测数据 |
|---|---|
| 全新 TLS 握手慢 | 放宽到 25s 后 **20/20 成功**，但最慢一次用了 **5.9s** |
| → 而 PTB 默认 `connect_timeout=5s` | 所以那次"失败"其实只是**慢了一点被掐断** |
| 并发新建连接会雪崩 | 同样 8s 超时下：并发 1 → **8/8**，并发 4 → **16/16**，并发 8 → 14/24，并发 40 → **4/40** |
| 复用已有连接又快又稳 | **25/25 成功，中位 0.26s**（新建连接要 0.73～2.0s，慢约 10 倍） |

**结论：主因是并发新建 TLS 连接，以及 PTB 默认的超时/连接池对高延迟链路太激进。**

> 我一度以为是"超时给小了"，但对照实验显示：40 并发下把超时从 5s 加到 30s，
> 失败数反而从 2/40 涨到 32/40 —— 那是被并发混淆了因果。控制变量后才定位到真正的变量。

### PTB 默认值 vs 本项目配置

| 参数 | PTB 默认 | 本项目 | 为什么 |
|---|---|---|---|
| `connection_pool_size` | **1** | 8 | 默认所有请求共用 1 条连接 |
| `connect_timeout` | **5s** | 20s | 握手实测最慢 5.9s |
| `read_timeout` | 5s | 30s | 长轮询需大于 poll timeout（已设 40s） |
| `write_timeout` | **5s** | 120s | 上传几 MB 图片远不够 |
| `pool_timeout` | **1s** | 30s | 默认等 1 秒就放弃，太短 |
| `media_write_timeout` | 20s | 300s | 媒体上传专用 |
| `concurrent_updates` | — | 4 | 避免同时下载/上传挤爆链路 |

另外做到：

1. **上传串行化**（`_upload_lock`）—— 并发大文件传输实测失败率从 ~100% 掉到 ~15%，所以同一时刻只发一个文件。
2. **自动重试 + 退避抖动** —— 下载 3 次、上传 4 次，指数退避加随机抖动，避免重试时再次冲垮链路。
3. **尊重 `RetryAfter`** —— 被限流时按 Telegram 指定的时间等待。

### ⚠️ 一个诚实的取舍

上传重试**可能造成极少数情况下重复发送**：如果请求其实已在服务端成功、只是响应超时，
重试就会再发一次。我选择"可能收到两次"而不是"什么都收不到"。
如果你更在意不重复，把 `bot.py` 里上传的 `attempts=4` 改成 `attempts=1`。

### 最有效的缓解：自己搭代理

如果链路长期不稳，比调参数更有效的是让 Bot 走本机代理（PTB 原生支持）：

```bash
export TELEGRAM_PROXY="http://127.0.0.1:7890"   # HTTP 代理，开箱可用
export TELEGRAM_PROXY="socks5://127.0.0.1:1080" # SOCKS5 需额外装依赖，见下
```

代码已支持：设置该变量即自动走代理，也会识别 `HTTPS_PROXY` / `ALL_PROXY`。
代理 URL 里的密码在日志中会被打码。

> **SOCKS5 需要额外依赖**（实测：不装会报
> `RuntimeError: To use Socks5 proxies, PTB must be installed via pip install "python-telegram-bot[socks]"`）：
>
> ```bash
> .venv/bin/python -m pip install "httpx[socks]"
> ```
>
> HTTP/HTTPS 代理**不需要**这个依赖。

注意本项目**不包含**代理服务，需要你自己准备。

## 测试

```bash
.venv/bin/python -m pip install pytest
.venv/bin/python -m pytest tests/ -v
```

若已安装 `node`，会自动执行与原 JavaScript 的差分测试；未安装则跳过。
参考 HTML 的路径可用环境变量覆盖：

```bash
REFERENCE_HTML=/path/to/小番茄图片混淆工具.html .venv/bin/python -m pytest tests/ -v
```

## 已知限制

| 限制 | 说明 |
|---|---|
| 不能用「照片」发送 | 见上文，这是最主要的坑 |
| 输入已被 JPEG 压缩 | 原工具本身就输出 JPEG 0.95，是有损的；Bot 输出 PNG 保证**不再**增加损失，但无法还原输入之前已丢失的信息 |
| 下载上限 20 MB | Telegram Bot API 的 `getFile` 限制 |
| 大图较慢 | 纯 Python 生成希尔伯特曲线是瓶颈：1080p 首次约 1 秒，12 MP 首次约 5 秒；曲线索引按宽高做了 LRU 缓存，相同尺寸复用后约 0.2 秒 |
| 像素数上限 50 MP | 先读文件头判断尺寸再解码，防止构造的压缩文件被解压成超大画布 |
| 状态不持久 | `/encrypt` 模式切换存在内存里，重启后回到默认解密 |
| 单实例轮询 | 用 `run_polling`，同一 token 不能同时跑多个实例 |

## 文件结构

```
image_confuse.py        算法核心：gilbert2d、置换、加解密、PIL 封装
bot.py                  Telegram Bot：接收 document、解混淆、回传 PNG
net.py                  网络容错：超时/连接池调优、重试退避、代理、上传串行化
healthcheck.py          容器健康检查（真的调一次 getMe）
cli.py                  命令行工具，批量加/解密
tests/test_algorithm.py 算法测试，含与原 JavaScript 的差分验证
tests/test_bot.py       Bot 处理流程与 CLI 测试
tests/test_net.py       网络容错层测试
Dockerfile / .dockerignore
docker-compose.yml / .env.example
requirements.txt        依赖
```

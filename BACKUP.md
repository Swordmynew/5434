# 完整服务器备份

[下载完整备份：backup-2026-10-28](https://github.com/Swordmynew/5434/releases/tag/backup-2026-10-28)

备份来源：`/root/autodl-tmp/5434`。备份日期：2026-10-28（Asia/Shanghai）。

备份包含整个目录中的源码、配置、测试、文档、原始数据集、全部模型检查点、实验结果、虚拟环境、缓存和隐藏文件，并保留目录结构、权限及符号链接。

- 原目录：15,519 个普通文件，21,757,754,461 字节。
- 压缩归档：16,012,268,789 字节，共 9 个分卷。
- Release 共上传 13 个附件：9 个分卷、还原说明、SHA-256 校验清单、源文件清单和归档目录清单。
- 上传完成后已逐一核对全部附件的大小及 GitHub 返回的 SHA-256。

## 下载与还原

下载 Release 中全部 `5434-full.tar.zst.part-000` 至 `5434-full.tar.zst.part-008`，以及 `RESTORE.md`、`SHA256SUMS`、`source-manifest.json` 和 `archive-list.txt`，放在同一个目录。

GitHub 页面自动生成的 **Source code (zip)** 和 **Source code (tar.gz)** 仅包含 Git 管理的源码；完整服务器目录保存在上述 9 个备份分卷中。

Linux 中安装 GNU tar 和 zstd 后，在一个有足够空间的新目录中执行：

```bash
sha256sum -c SHA256SUMS
mkdir -p restored
cat 5434-full.tar.zst.part-* | zstd -dc | tar --acls --xattrs --numeric-owner -xpf - -C restored
```

还原结果为 `restored/5434/`。如需保留数字用户/组身份，以 root 身份解压。解压需额外预留至少 22 GB 空间。

虚拟环境和缓存按原样保存；其中部分符号链接指向项目目录之外的 Python 解释器或系统依赖。更换服务器时，仍需具备相应 Python、CUDA 和系统依赖，或根据 `pyproject.toml` 重建环境。

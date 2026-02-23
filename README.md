# MLIPPER

MLIPPER 是一個使用 CUDA 加速的工具，用來讀取參考比對（MSA）與參考樹（Newick），並在 GPU 上進行樹相關的似然計算／查詢準備。

## Build

在專案根目錄：

```bash
make
```

常用 Make 參數（可用環境變數覆寫）：

- `CUDA_HOME`（預設 `/usr/local/cuda-12`）
- `PLL_INC_DIR` / `PLL_LIB_DIR`（預設 `/usr/local/include`、`/usr/local/lib`，用於 `libpll`）
- `DEBUG=1`：debug build（`make debug` 也可以）
- `USE_DOUBLE=1`：使用 double precision

清除：

```bash
make clean
```

## Quick start

你可以直接用 `run.sh`（從專案根目錄執行）：

```bash
./run.sh
```

`run.sh` 目前等價於：

```bash
./MLIPPER \
  --tree-alignment ./data/aln.fasta \
  --query-alignment ./data/query.fasta \
  --tree ./data/ref.tre \
  --states 4 \
  --subst-model GTR \
  --ncat 4 \
  --alpha 0.3 \
  --pinv 0.0 \
  --freqs 0.25,0.25,0.25,0.25 \
  --rates 1.0,1.0,1.0,1.0,1.0,1.0 \
  --rate-weights 0.25,0.25,0.25,0.25
```

## CLI options

查看完整參數：

```bash
./MLIPPER --help
```

重點參數（節錄）：

- `--tree-alignment <FILE>`：參考樹的 MSA（FASTA）
- `--query-alignment <FILE>`：查詢序列的 MSA（FASTA；可省略，省略時會用 `--tree-alignment`）
- `--tree <FILE>`：參考樹拓樸（Newick 檔）
- `--tree-newick <TEXT>`：直接用字串提供 Newick（與 `--tree` 互斥）
- `--states <INT>`：states 數量（DNA 常用 4）
- `--subst-model <TEXT>`：目前主要支援 `GTR`
- `--ncat <INT>` / `--alpha <FLOAT>`：Gamma rate categories 與 alpha
- `--freqs <LIST>`：平衡頻率（可用逗號分隔）
- `--rates <LIST>`：GTR 六個 rate（rAC,rAG,rAT,rCG,rCT,rGT；可用逗號分隔）
- `--rate-weights <LIST>`：各 rate category 權重（可用逗號分隔）
- `--no-per-rate-scaling`：關閉 per-rate scaling

## Notes (generated files)

- 執行時如果你有提供 `--tree` 或 `--tree-newick`，程式目前會在「當前工作目錄」寫出 `./tree.nwk`（作為 `libpll` 解析 Newick 的暫存檔）。這個檔案已被 `.gitignore` 忽略（`*.nwk`）。


# penet5: HTCL 实验（Head-Tail Cooperative Learning）

## 实验信息
- **方法**: HTCL (Head-Tail Cooperative Learning Network)
- **论文**: Wang et al., Image and Vision Computing 2024
- **代码**: https://github.com/wanglei0618/HTCL
- **验证数据**: ⚠️ 无公开PE-NET数据，但代码包含PE-NET支持

## ⚠️ CB-Loss正交性警告
HTCL config中有 `num_beta: 0.9999`，说明内部已包含class-balanced reweighting。
如果实验达标，需要单独测试去掉num_beta后的效果，确认CB-Loss的独立贡献。

## 运行方式
```bash
bash experiments/round3/run_htcl.sh
```

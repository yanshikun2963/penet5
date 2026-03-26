#!/bin/bash
# ================================================================
#  penet5: HTCL 实验（Head-Tail Cooperative Learning）
#  预期结果: 未验证，但架构上专门优化F@K（R和mR的平衡）
#  预计时间: 搭建1h + 训练12-16h
#  论文: Wang et al., Image and Vision Computing 2024
#  代码: github.com/wanglei0618/HTCL
# ================================================================
set -e

echo "================================================================"
echo "  Step 0: 环境准备"
echo "================================================================"

# 你需要修改以下路径
WORK_DIR="/path/to/your/workdir"
PENET_DIR="${WORK_DIR}/penet5"
DATASET_DIR="${PENET_DIR}/datasets/vg"
GLOVE_DIR="${DATASET_DIR}"
DETECTOR_CKPT="${PENET_DIR}/checkpoints/pretrained_faster_rcnn/model_final.pth"
GPU_ID=0

# ================================================================
echo "  Step 1: 克隆 HTCL 代码库"
echo "================================================================"

cd ${WORK_DIR}
if [ ! -d "HTCL" ]; then
    git clone https://github.com/wanglei0618/HTCL.git
    cd HTCL
    python setup.py build develop
else
    echo "HTCL already exists, skipping clone"
    cd HTCL
fi

# ================================================================
echo "  Step 2: 修改配置路径"
echo "================================================================"

# 修改 HTCL_base_opts.yaml 中的路径
OPTS_FILE="./maskrcnn_benchmark/HTCL/HTCL_base_opts.yaml"
if [ -f "$OPTS_FILE" ]; then
    # 备份原始文件
    cp ${OPTS_FILE} ${OPTS_FILE}.bak

    # 用sed替换关键路径
    sed -i "s|GLOVE_DIR:.*|GLOVE_DIR: ${GLOVE_DIR}|g" ${OPTS_FILE}
    sed -i "s|PRETRAINED_DETECTOR_CKPT:.*|PRETRAINED_DETECTOR_CKPT: ${DETECTOR_CKPT}|g" ${OPTS_FILE}

    echo "配置文件已更新: ${OPTS_FILE}"
    echo "关键参数:"
    grep "GLOVE_DIR\|PRETRAINED_DETECTOR_CKPT\|num_beta\|Classifier_Finetuning\|Feats_resampling" ${OPTS_FILE}
else
    echo "❌ 配置文件未找到: ${OPTS_FILE}"
    exit 1
fi

# ================================================================
echo "  Step 3: 链接数据集"
echo "================================================================"

cd ${WORK_DIR}/HTCL
ln -sf ${DATASET_DIR}/../.. ./datasets 2>/dev/null || true
ln -sf ${PENET_DIR}/checkpoints/pretrained_faster_rcnn ./checkpoints/pretrained_faster_rcnn 2>/dev/null || true

# ================================================================
echo "  Step 4: 预训练检查"
echo "================================================================"

echo -n "数据集: "
ls ./datasets/vg/VG-SGG-with-attri.h5 2>/dev/null && echo "✅" || echo "❌"

echo -n "Detector: "
ls ${DETECTOR_CKPT} 2>/dev/null && echo "✅" || echo "❌"

echo -n "HTCL训练入口: "
ls ./tools/HTCL_main.py 2>/dev/null && echo "✅" || echo "❌"

echo -n "PENET_HTCL predictor: "
grep -c "class PENET_HTCL" ./maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py && echo "✅" || echo "❌"

echo -n "GPU: "
python3 -c "import torch; print(f'✅ {torch.cuda.get_device_name(${GPU_ID})}')" 2>/dev/null || echo "❌"

# ================================================================
echo ""
echo "  ⚠️  重要提醒"
echo "================================================================"
echo ""
echo "HTCL config里有 num_beta: 0.9999 —— 这和CB-Loss的β相同！"
echo "这意味着HTCL内部已经包含了class-balanced reweighting。"
echo ""
echo "如果HTCL达标(mR@50≥35, R@50≥58)，你需要检查："
echo "  1. 把 num_beta 设成 0（关闭reweighting）再跑一次"
echo "     看没有reweighting的HTCL能不能到35"
echo "  2. 如果能 → HTCL本身就是组件二，CB-Loss是独立的组件一"
echo "  3. 如果不能 → HTCL的增益来自reweighting，和CB-Loss有重叠"
echo ""

# ================================================================
echo "  Step 5: 开始训练"
echo "================================================================"

export CUDA_VISIBLE_DEVICES=${GPU_ID}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256

OUTPUT_BASE="./checkpoints/HTCL/PENET_HTCL_preds"
mkdir -p ${OUTPUT_BASE}
mkdir -p ${OUTPUT_BASE}/cls_ft
mkdir -p ./feats/HTCL/PENET_HTCL_preds

nohup python3 -u tools/HTCL_main.py \
  --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
  --my_opts "./maskrcnn_benchmark/HTCL/HTCL_base_opts.yaml" \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
  MODEL.ROI_RELATION_HEAD.PREDICTOR PENET_HTCL \
  OUTPUT_DIR ${OUTPUT_BASE} \
  OUTPUT_DIR_feats ./feats/HTCL/PENET_HTCL_preds \
  CLASSIFIER_OUTPUT_DIR ${OUTPUT_BASE}/cls_ft \
  SOLVER.PRE_VAL False \
  2>&1 | tee ${OUTPUT_BASE}/train.log &

echo ""
echo "================================================================"
echo "  HTCL 训练已启动！PID: $!"
echo "  监控: tail -f ${OUTPUT_BASE}/train.log"
echo "  注意: HTCL有3个阶段，总计约12-16h"
echo "    Phase 1: PE-NET训练 (60000 iter)"
echo "    Phase 2: 特征提取+重采样"
echo "    Phase 3: 分类器微调 (100000 iter, batch=64)"
echo "================================================================"

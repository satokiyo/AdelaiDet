#OMP_NUM_THREADS=1 python tools/train_net.py \
#export DETECTRON2_DATASETS=/media/HDD2/20211104_panet/sample_data/ROI/cytology_sample_x20_512x512/train/sample_data/coco
#    register_coco_instances("my_dataset", {}, "/media/HDD2/20211104_panet/sample_data/ROI/cytology_sample_x20_512x512/train/sample_data/HSILsample01/annotations/HSILsample01.json", "/media/HDD2/20211104_panet/sample_data/ROI/cytology_sample_x20_512x512/train/sample_data/HSILsample01")

python3 tools/train_net.py \
    --config-file swin_transformer/configs/SwinT/mask_rcnn_swint_T_FPN_3x_mydataset.yaml \
    --num-gpus 1 \
    OUTPUT_DIR ./out/mask_rcnn_swint_T_FPN_3x_pseudoLabel

#python3 tools/train_net.py \
#    --config-file configs/SOLOv2/R50_3x.yaml \
#    --eval-only \
#    --num-gpus 1 \
#    OUTPUT_DIR ./out/SOLOv2_R50_3x \
#    MODEL.WEIGHTS ./out/SOLOv2_R50_3x/model_final.pth

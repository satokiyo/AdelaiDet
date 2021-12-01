python3 demo/demo.py \
    --config-file swin_transformer/configs/SwinT/mask_rcnn_swint_T_FPN_3x_mydataset.yaml \
    --input /media/HDD2/20211104_panet/sample_data/ROI/cytology_sample_x20_512x512/train/sample_data/coco \
    --output /media/HDD2/20211111_solov2/sample_out/swint/pred_HSIL \
    --opts MODEL.WEIGHTS ./out/mask_rcnn_swint_T_FPN_3x/model_0041999.pth \
           MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.5

    #--input /media/HDD2/20211104_panet/sample_data/kaggle_2018/image/stage1_test \
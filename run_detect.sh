python3 demo/demo.py \
    --config-file configs/SOLOv2/R50_3x.yaml \
    --input /media/HDD2/20211104_panet/sample_data/kaggle_2018/image/stage1_test \
    --output /media/HDD2/20211111_solov2/sample_out/solov2/pred \
    --opts MODEL.WEIGHTS ./out/SOLOv2_R50_3x/model_0011969.pth \
           MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.5 
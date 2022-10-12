#!/bin/bash
python inference_unet.py -img_dir ./test_imgs -model_path ./models3/model-0/model-0_best.pt -out_viz_dir ./test_img33
python inference_unet.py -img_dir ./test_imgs -model_path ./models3/model-1/model-1_best.pt -out_viz_dir ./test_img4
python inference_unet.py -img_dir ./test_imgs -model_path ./models3/model-2/model-2_best.pt -out_viz_dir ./test_img5
python inference_unet.py -img_dir ./test_imgs -model_path ./models3/model-0/model-0_epoch_40.pt -out_viz_dir ./test_img6
python inference_unet.py -img_dir ./test_imgs -model_path ./models3/model-1/model-1_epoch_40.pt -out_viz_dir ./test_img7
python inference_unet.py -img_dir ./test_imgs -model_path ./models3/model-2/model-2_epoch_40.pt -out_viz_dir ./test_img8
python inference_unet.py -img_dir ./test_imgs -model_path ./models3/model-0/model-0_epoch_30.pt -out_viz_dir ./test_img9
python inference_unet.py -img_dir ./test_imgs -model_path ./models3/model-1/model-1_epoch_30.pt -out_viz_dir ./test_img10
python inference_unet.py -img_dir ./test_imgs -model_path ./models3/model-2/model-2_epoch_30.pt -out_viz_dir ./test_img11
python inference_unet.py -img_dir ./test_imgs -model_path ./models3/model-0/model-0_epoch_10.pt -out_viz_dir ./test_img12
python inference_unet.py -img_dir ./test_imgs -model_path ./models3/model-1/model-1_epoch_10.pt -out_viz_dir ./test_img13
python inference_unet.py -img_dir ./test_imgs -model_path ./models3/model-2/model-2_epoch_10.pt -out_viz_dir ./test_img13

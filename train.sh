 #python multigpu_train.py --gpu_list=0,1 --input_size=512 --batch_size_per_gpu=14 --checkpoint_path=./log/ \
 #--text_scale=512 --training_data_path=./ICDAR --geometry=RBOX --learning_rate=0.0001 --num_readers=24 \
 #--pretrained_model_path=./resnet_v1_50.ckpt

 #python multigpu_train.py --gpu_list=0,1 --input_size=512 --batch_size_per_gpu=14 --checkpoint_path=./log/ \
 #--text_scale=512 --training_data_path=./ICDAR --geometry=RBOX --learning_rate=0.0001 --num_readers=24 \
 #--pretrained_model_path=./resnet_v2_101.ckpt

 python multigpu_train.py --gpu_list=0,1 --input_size=512 --batch_size_per_gpu=14 --checkpoint_path=./exhibition_log/ \
 --text_scale=512 --training_data_path=./exhibition --geometry=RBOX --learning_rate=0.0001 --num_readers=24 \
 --pretrained_model_path=./resnet_v1_101.ckpt

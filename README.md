# Bidirectional Pyramid Networks for Semantic Segmentation
by Nie, D., Xue, J. and Ren, X., details can be found [here](https://openaccess.thecvf.com/content/ACCV2020/html/Nie_Bidirectional_Pyramid_Networks_for_Semantic_Segmentation_ACCV_2020_paper.html) 

## Introduction
This repository is build for the proposed Bidirectional Pyramid Networks (BPNet), which contains full training and testing code on several segmentation datasets. 

![] (https://github.com/ginobilinie/BPNet/raw/master/img/arch1.png)

## Usage
1. Requirement:

   - Hardware: tested with RTX 2080 TI (11G).

   - Software: tested with PyTorch 1.2.0, Python3.7, CUDA 10.0, tensorboardX, Ninja, tqdm, Easydict
   
   - Anaconda is strongly recommended


2. Clone the repository:
<pre> 
   git clone https://github.com/ginobilinie/BPNet.git 
</pre>

3. How to Train
   - create the config file of dataset:train.txt, val.txt, test.txt
   
   file structure：(split with tab)
   <pre>
   path-of-the-image   path-of-the-groundtruth
   </pre>
   
   - modify the config.py according to your requirements
   
   - train a network:
<pre>
   export NGPUS=8
   python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py  
</pre>

4. Test

   - inference
   <pre>
     python eval -e epoch_idx -d device_idx [--verbose ] [--show_image] [--save_path Pred_Save_Path]

   </pre>

5. Visualization

6. Other Resources

   Resources: GoogleDrive [LINK]() contains pretrained models and some share models. 

## Cite
<pre>
@inproceedings{nie2020bidirectional,
  title={Bidirectional Pyramid Networks for Semantic Segmentation},
  author={Nie, Dong and Xue, Jia and Ren, Xiaofeng},
  booktitle={Proceedings of the Asian Conference on Computer Vision},
  year={2020}
}
</pre>

## Thanks

Our work uses part of codes from https://github.com/ycszen/TorchSeg and https://github.com/MendelXu/ANN. Thanks for your great work!

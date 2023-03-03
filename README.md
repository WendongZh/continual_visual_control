### Predictive Experience Replay for Continual Visual Control and Forecasting

Code for Predictive Experience Replay for Continual Visual Control and Forecasting.

This project is for our new approach on continual visual control which is submitted to the journal for possible publication. This work is an extension version of our previous work [CPL (CVPR'22)](https://openaccess.thecvf.com/content/CVPR2022/html/Chen_Continual_Predictive_Learning_From_Videos_CVPR_2022_paper.html). If you have any questions, feel free to make issues. Thanks for your interests!

### Introduction:
Briefly speaking, in this work, we extend our previous work on continual visual forecasting to address continual visual control tasks based on the model-based RL approach Dreamer. The key assumption is that an ideal world model can provide a non-forgetting environment simulator, which enables the agent to optimize the policy in a multi-task learning manner based on the imagined trajectories from the world model. We summarize the main contributions of this paper as follows:
1) We present the mixture world model architecture to learn task-specific Gaussian priors, which lays a foundation for the continual learning of spatiotemporal dynamics.
2) We introduce the predictive experience replay for memory-efficient data rehearsal, which is the key to overcoming the catastrophic forgetting for the world model.
3) On the basis of the above two techniques, we make a pilot study of MBRL for continual visual control tasks, and propose the exploratory-conservative behavior learning method to improve the value estimation learned from replayed data.
4) We show extensive experiments to demonstrate the superiority of our approach over the most advanced methods in visual control and forecasting. The results also indicate that learning a non-forgetting world model is a cornerstone to continual RL.

<p align='center'>  
  <img src='https://github.com/WendongZh/continual_visual_control/blob/main/save_img/pami_githubpng.PNG' width='870'/>
</p>

## Prerequisites
- [Deepmind Control Suite](https://github.com/deepmind/dm_control)
- [Meta-World](https://github.com/Farama-Foundation/Metaworld)
- [Mujuco](https://github.com/deepmind/mujoco)

## Benchmark
We evaluate our

## Getting Strated
Since our approach can be applied for both deterministic and probabilistic image inpainting, so we seperate the codes under these two setups in different files and each file contains corresponding training and testing commonds.

For all setups, the common pre-preparations are list as follows:

1) Download the pre-trained models and copy them under ./checkpoints directory. 

2) (For training) Make another directory, e.g ./pretrained_ASL, and download the weights of [TResNet_L](https://github.com/Alibaba-MIIL/ASL/blob/main/MODEL_ZOO.md) pretrained on OpenImage dataset to this directory.

3) Install torchlight
```bash
cd ./torchlight
python setup.py install
```

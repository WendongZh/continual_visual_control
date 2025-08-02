# Continual Visual Reinforcement Learning with A Life-Long World Model [ECML 2025]

Code for Continual Visual Reinforcement Learning with A Life-Long World Model. [Paper on arxiv](https://arxiv.org/abs/2303.06572)

This work is an extension version of our previous work [CPL (CVPR'22)](https://openaccess.thecvf.com/content/CVPR2022/html/Chen_Continual_Predictive_Learning_From_Videos_CVPR_2022_paper.html). If you have any questions, feel free to make issues. Thanks for your interests!

### Introduction:
Briefly speaking, in this work, we extend our previous work on continual visual forecasting to address continual visual control tasks based on the model-based RL approach Dreamer. The key assumption is that an ideal world model can provide a non-forgetting environment simulator, which enables the agent to optimize the policy in a multi-task learning manner based on the imagined trajectories from the world model. We summarize the main contributions of this paper as follows:
1) We present the mixture world model architecture to learn task-specific Gaussian priors, which lays a foundation for the continual learning of spatiotemporal dynamics.
2) We introduce the predictive experience replay for memory-efficient data rehearsal, which is the key to overcoming the catastrophic forgetting for the world model.
3) On the basis of the above two techniques, we make a pilot study of MBRL for continual visual control tasks, and propose the exploratory-conservative behavior learning method to improve the value estimation learned from replayed data.
4) We show extensive experiments to demonstrate the superiority of our approach over the most advanced methods in visual control and forecasting. The results also indicate that learning a non-forgetting world model is a cornerstone to continual RL.


## Prerequisites
- [Deepmind Control Suite](https://github.com/deepmind/dm_control)
- [Meta-World](https://github.com/Farama-Foundation/Metaworld)
- [Mujuco](https://github.com/deepmind/mujoco)

## Benchmark
We evaluate our approach on tasks collected from two different RL platforms: DMC and Meta-World. In our paper, the continual tasks on DMC are Walker-walk, Walker-uphill, Walker-downhill, and Walker-nofoot. The continual tasks on Meta-World are Window-open, Button-press, Hammer, and Assembly. 

Particularly, the last three tasks on the DMC benchmarks are mannually designed and you need to copy the files in ./metaworld_walker_xml to the DMC directory in your conda environment. These files do not affect the original tasks in DMC benchmarks. **However, I STRONGLY suggest checking these files with your original files to make sure that they are belong to the same DMC version.**

## Getting Strated

1) Copy all files in ./metaworld_walker_xml to the DMC directory in your conda environment, such as /home/.conda/envs/your_env_name/lib/python3.7/site-packages/dm_control/suite/

2) Training command on the DMC:  
```bash
EGL_DEVICE_ID=7 CUDA_VISIBLE_DEVICES=7 python dreamer.py --configs defaults dmc
```
The default training order in configs.yaml is walk->uphill->downhill->nofoot. You can change it to test other orders.

3) Training command on the Meta-World:  
```bash
CUDA_VISIBLE_DEVICES=7 python dreamer.py --configs defaults metaworld
```
You should also mannually modifiy the gpu id in the MetaWorld class in the wrappers.py file (line 336 and 350) to assign the gpu for rendering.

## Appreciation
The codes refer to the implemention of [dreamer-torch](https://github.com/jsikyoon/dreamer-torch). Thanks for the authors！

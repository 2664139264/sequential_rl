#### 说明 ####

依赖 fork 后的 stable_baselines3，rl_zoo3，stable_baselines3_contrib 三个库的 local 分支
这些库为本库的子模块，需要独立 push 来修改这些库的 local 分支
在服务器上 pip install -e . 这些库


#### IDEA ####

1. 差分状态作为状态的表示的意义，对观测序列施加线性算符（微分、积分、掩码？）


#### 索引备查 ####

0. 训练启动入口                     rl-baselines3-zoo/exp_manager.py
1. 注册部分可观测环境               rl-baselines3-zoo/rl_zoo3/import_envs.py
2. 自定义 Wrapper                  rl-baselines3-zoo/rl_zoo3/wrappers.py
3. 算法超参数                      rl-baselines3-zoo/hyperparams
4. RNN 网络定义                    stable-baselines3-contrib/sb3_contrib/common/recurrent/policies.py
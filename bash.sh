# Set up cmake
mkdir build && mkdir data
cmake -S . -B build
# Build rocksdb
cmake --build build
# 开启/关闭虚拟环境
conda activate camal_env
conda deactivate

# 删除旧的临时目录
rm -rf /tmp/level_cost
rm -rf /tmp/level_optimizer

python3 

# tmux会话 新建 + 重新进入
tmux new -s lsmacache
tmux attach -t lsmacache

# 删除会话
# 直接kill指定会话
tmux kill-session -t lsmacache
# 在tmux会话中使用快捷键 
exit  # ✅✅✅ 使用这个方式正常退出

sudo lsof +L1 /data
du -h --max-depth=1 /data
df -h


# 重定向标准输出和标准错误（全部输出）
python script.py > /home/ubuntu/projects/LSMCACHE/data/output.txt 2>&1
# /home/ubuntu/projects/LSMCACHE/data/output.txt

python3 /home/ubuntu/projects/LSMCACHE/lsmcache/sampling/lsmwrite_cache_xgb.py  > /home/ubuntu/projects/LSMCACHE/data/output.txt 2>&1
python3 /home/ubuntu/projects/LSMCACHE/lsmcache/train/lsmwrite_cache_cost_xgb.py > /home/ubuntu/projects/LSMCACHE/data/output.txt 2>&1
python3 /home/ubuntu/projects/LSMCACHE/lsmcache/optimizer/lsmwrite_cache_xgb_optimizer.py > /home/ubuntu/projects/LSMCACHE/data/output.txt 2>&1

python3 memory_tuner/main.py > data/output.txt 2>&1
python3 motivating_exp/main.py > data/output.txt 2>&1

# test
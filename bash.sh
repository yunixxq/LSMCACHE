# Set up cmake
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

# tmux会话
tmux new -s lsmacache
tmux attach -t lsmacache

# 删除会话
# 直接kill指定会话
tmux kill-session -t lsmacache
# 在tmux会话中使用快捷键
exit
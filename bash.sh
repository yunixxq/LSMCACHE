# Set up cmake
cmake -S . -B build
# Build rocksdb
cmake --build build
# 开启/关闭虚拟环境
conda activate camal_env
conda deactivate

# 删除旧的临时目录
rm -rf /tmp/level_cost

python3 
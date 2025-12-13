#include "data_generator.hpp"
#include "zipf.hpp"
#include "scrambled_zipfian_generator.h"
#include "uniform_generator.h"
using namespace ycsbc;
Uniform::Uniform(int max)
{
    this->dist = std::uniform_int_distribution<int>(0, max);
}

int Uniform::gen(std::mt19937 &engine)
{
    return this->dist(engine);
}

Zipf::Zipf(int max)
{
    this->dist = opencog::zipf_distribution<int, double>(max);
}

int Zipf::gen(std::mt19937 &engine)
{
    return this->dist(engine);
}

RandomGenerator::RandomGenerator(int seed)
{
    this->seed = seed;
    this->engine.seed(this->seed);
    srand(this->seed);
    this->dist = std::uniform_int_distribution<int>(0, KEY_DOMAIN);
}

std::pair<std::string, std::string> DataGenerator::gen_kv_pair(size_t kv_size)
{
    std::string key = this->gen_key();
    assert(key.size() < kv_size && "Requires larger key size");
    size_t value_size = kv_size - key.size();
    std::string value = this->gen_val(value_size);

    return std::pair<std::string, std::string>(key, value);
}

std::pair<std::string, std::string> DataGenerator::gen_new_kv_pair(size_t kv_size)
{
    std::string key = this->gen_new_dup_key();
    assert(key.size() < kv_size && "Requires larger key size");
    size_t value_size = kv_size - key.size();
    std::string value = this->gen_val(value_size);

    return std::pair<std::string, std::string>(key, value);
}

// ✅xxq新增，生成符合Zipfian分布规律的已存在key的kv对
std::pair<std::string, std::string> DataGenerator::gen_existing_kv_pair(size_t kv_size)
{
    std::string key = this->gen_existing_key();
    assert(key.size() < kv_size && "Requires larger key size");
    size_t value_size = kv_size - key.size();
    std::string value = this->gen_val(value_size);

    return std::pair<std::string, std::string>(key, value);
}

std::string RandomGenerator::gen_key()
{
    return std::to_string(this->dist(this->engine));
}

std::string RandomGenerator::gen_val(size_t value_size)
{
    std::string value = std::string(value_size, 'a');

    return value;
}

KeyFileGenerator::KeyFileGenerator(std::string key_file, int offset, int num_keys, int seed, std::string mode)
{
    this->mode = mode;
    this->engine = std::mt19937(seed);
    this->read_file(key_file, offset, num_keys);
    if (mode == "uniform")
    {
        this->dist_existing = new Uniform(offset);
        this->dist_new = new Uniform(num_keys);
    }
    else
    {
        this->dist_existing = new Zipf(offset);
        this->dist_new = new Zipf(num_keys);
    }
}

KeyFileGenerator::~KeyFileGenerator()
{
    delete this->dist_new;
    delete this->dist_existing;
}

void KeyFileGenerator::read_file(std::string key_file, int offset, int num_keys)
{
    spdlog::info("Reading in key_file {}", key_file);
    std::ifstream fid(key_file, std::ios::in | std::ios::binary);
    if (!fid)
    {
        spdlog::warn("Error opening key file {}", key_file);
    }

    this->existing_keys.resize(offset);
    fid.read(reinterpret_cast<char *>(this->existing_keys.data()), offset * sizeof(int));
    // for (int i = 0; i < 10; i++)
    //     spdlog::info(existing_keys[i]);
    // spdlog::info("=====================================");
    this->keys.resize(num_keys);
    fid.read(reinterpret_cast<char *>(this->keys.data()), num_keys * sizeof(int));
    // for (int i = 0; i < 10; i++)
    //     spdlog::info(keys[i]);
    this->key_gen = this->keys.begin();
    spdlog::debug("Size of exisiting, new : {}, {}", this->existing_keys.size(), this->keys.size());

    fid.close();
}

std::string KeyFileGenerator::gen_key()
{
    std::string key = std::to_string(*this->key_gen);
    this->key_gen++;

    return key;
}

std::string KeyFileGenerator::gen_val(size_t value_size)
{
    return std::string(value_size, 'a');
}

std::string KeyFileGenerator::gen_new_dup_key()
{
    std::string key = std::to_string(this->keys[this->dist_new->gen(this->engine) - 1]);

    return key;
}

std::string KeyFileGenerator::gen_existing_key()
{
    return std::to_string(this->existing_keys[this->dist_existing->gen(this->engine) - 1]);
}

// ✅重点关注YCSB的实现 num_keys = N; mode = uniform / zipfian
YCSBGenerator::YCSBGenerator(int num_keys, std::string mode, double zipfian_skewness)
{
    // 1. 生成key列表
    std::vector<int> v;
    for (int i = 0; i < num_keys; ++i)
    {
        v.push_back(i);
    }
    this->existing_keys = v; // 初始化 existing_keys 为 [0, 1, 2, ..., N-1]
    this->keys = v;
    this->key_gen = this->existing_keys.begin();

    // 2. 根据mode创建分布生成器
    if (mode == "uniform")
    {
        // 均匀分布：每个key被选中的概率相等(未使用)
        this->dist_existing = new UniformGenerator(0, num_keys - 1);
        this->dist_new = new UniformGenerator(0, num_keys - 1);  // 实际上并未使用
    }
    else
    {
        // ✅(打乱版)Zipfian分布：某些热点key被选中的概率更高
        this->dist_existing = new ScrambledZipfianGenerator(0, num_keys-1, zipfian_skewness);
        this->dist_new = new ScrambledZipfianGenerator(0, num_keys-1, zipfian_skewness); // 实际上并未使用
    }
}

// "0", "1", ..., "N-1"(循环)
std::string YCSBGenerator::gen_key()
{
    std::string key = std::to_string(*this->key_gen);
    this->key_gen++;
    if (this->key_gen == this->existing_keys.end())
        this->key_gen = this->existing_keys.begin();
    return key;
}

YCSBGenerator::~YCSBGenerator()
{
    delete this->dist_new;
    delete this->dist_existing;
}

// ✅生成空点查询的key
std::string YCSBGenerator::gen_new_dup_key()
{
    std::string key = std::to_string(this->existing_keys[this->dist_existing->Next()]);
    int last_char = int(key.at(key.length() - 1)); // 获取key的最后一个字符的ASCII值
    return key.substr(0, key.length() - 1) + (char)(last_char + 17); // '0'-'9'加上17后变成'A'-'J'，而existing_key中一定不存在字母
}

// ✅生成非空点查询的key
std::string YCSBGenerator::gen_existing_key()
{
    std::string key = std::to_string(this->existing_keys[this->dist_existing->Next()]);
    return key;
}

std::string YCSBGenerator::gen_val(size_t value_size)
{
    return std::string(value_size, 'a');
}

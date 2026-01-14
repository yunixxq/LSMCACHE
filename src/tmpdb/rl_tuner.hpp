#ifndef TMPDB_RL_TUNER_HPP
#define TMPDB_RL_TUNER_HPP

#include <vector>
#include <random>
#include <string>
#include <cmath>
#include <algorithm>
#include <deque>

namespace tmpdb {

constexpr int NUM_ACTIONS = 5;
constexpr int STATE_FEATURES = 6;
constexpr int FEATURE_DIM = STATE_FEATURES * NUM_ACTIONS + NUM_ACTIONS;

// 动作枚举 - 5个离散动作 - 比连续动作空间更稳定、更易收敛
// Large/Small可以让agent可以根据当前性能差距选择合适的调整幅度
enum class Action {
    DECREASE_LARGE = 0,   // alpha -= 2*step
    DECREASE_SMALL = 1,   // alpha -= step
    HOLD = 2,             // alpha 不变
    INCREASE_SMALL = 3,   // alpha += step
    INCREASE_LARGE = 4    // alpha += 2*step
};

// 动作转字符串
std::string action_to_string(Action action);

// 动作转alpha变化量
double action_to_delta(Action action, double step_size);

// RL状态
struct RLState {
    // 环境状态(Context) - 当前决策点
    double read_ratio;           // 当前workload读比例
    double current_alpha;        // 当前alpha值
    
    // 历史性能状态 - 提供性能趋势
    double H_cache_prev;         // 上一epoch命中率
    double H_cache_delta;        // 命中率变化
    double latency_prev;         // 上一epoch延迟 (ms)
    double io_prev;              // 上一epoch IO (KB/op)
    
    // 元信息
    int episode_count;           // 当前workload下的epoch计数
    
    RLState();
    void reset(double alpha, double read_ratio);
    void update_from_stats(double H_cache, double latency_ms, double io_kb_per_op,
                           double prev_H_cache);
};

// Q函数(线性函数逼近)
class QFunction {
public:
    QFunction();
    
    // 计算Q值
    double compute(const RLState& state, Action action) const;
    
    // 找最优动作
    Action best_action(const RLState& state) const;
    
    // Q-Learning更新
    void update(const RLState& prev_state, Action action, double reward,
                const RLState& curr_state, double learning_rate, double discount_factor);
    
    // 获取所有动作的Q值（用于调试）
    std::vector<double> get_all_q_values(const RLState& state) const;
    
    // 重置权重
    void reset();

private:
    std::vector<double> weights_;
    
    // 提取特征向量
    std::vector<double> extract_features(const RLState& state, Action action) const;
};

// UCB探索跟踪器
class UCBTracker {
public:
    explicit UCBTracker(double exploration_weight = 1.414);
    
    // 计算UCB值
    double ucb_value(Action action) const;
    
    // 更新统计
    void update(Action action, double reward);
    
    // 重置（workload变化时）
    void reset();
    
    // 获取总尝试次数
    int total_count() const { return total_count_; }
    
    // 获取某动作的平均奖励
    double avg_reward(Action action) const;

private:
    std::vector<int> action_counts_;
    std::vector<double> action_rewards_;
    int total_count_;
    double exploration_weight_;
};

// RL配置
struct RLConfig {
    double step_size = 0.05;           // alpha调整步长
    double learning_rate = 0.1;        // Q-Learning学习率
    double discount_factor = 0.9;      // 折扣因子
    double epsilon_start = 0.3;        // 初始探索率
    double epsilon_decay = 0.95;       // 探索率衰减
    double epsilon_min = 0.05;         // 最小探索率
    double ucb_c = 1.414;              // UCB探索系数
    double alpha_min = 0.05;           // alpha下限
    double alpha_max = 0.95;           // alpha上限
    double convergence_threshold = 0.005;  // H_cache变化阈值
};

// Epoch性能统计（简化版，用于RL内部）
struct EpochPerf {
    double H_cache;
    double latency_ms;
    double io_kb_per_op;
    double read_ratio;
    
    EpochPerf() : H_cache(0), latency_ms(0), io_kb_per_op(0), read_ratio(0) {}
    EpochPerf(double h, double lat, double io, double rr)
        : H_cache(h), latency_ms(lat), io_kb_per_op(io), read_ratio(rr) {}
};

// RL调优器主类
class RLTuner {
public:
    explicit RLTuner(const RLConfig& config);
    
    // 初始化（使用GBDT预测的alpha）
    void init(double initial_alpha, double initial_read_ratio);
    
    // 处理一个epoch的结果，返回新的alpha
    // 返回值: 新的alpha值
    double on_epoch_end(const EpochPerf& perf);
    
    // 处理workload漂移，使用新的GBDT预测alpha进行Jump Start
    void on_drift_detected(double new_predicted_alpha, double new_read_ratio);
    
    // 获取当前alpha
    double current_alpha() const { return state_.current_alpha; }
    
    // 获取当前探索率
    double current_epsilon() const { return epsilon_; }
    
    // 获取最后执行的动作
    Action last_action() const { return last_action_; }
    
    // 获取最后的奖励
    double last_reward() const { return last_reward_; }
    
    // 获取统计信息（用于日志）
    std::string get_stats_string() const;
    
    // 获取Q值信息（用于调试）
    std::vector<double> get_q_values() const;
    
    // 设置随机种子
    void set_seed(int seed);

    bool is_converged() const { return converged_; }

    // 重置收敛状态（drift时调用）
    // void reset_convergence() { 
    //     recent_rewards_.clear(); 
    //     converged_ = false; 
    // }

private:
    RLConfig config_;
    QFunction q_function_;
    UCBTracker ucb_tracker_;
    RLState state_;
    
    double epsilon_;
    bool has_prev_perf_;
    EpochPerf prev_perf_;
    Action last_action_;
    double last_reward_;
    int epoch_count_;
    
    std::deque<double> recent_rewards_;      // 最近N个epoch的reward
    int convergence_window_ = 3;             // 收敛判断窗口大小
    bool converged_ = false;                 // 是否已收敛

    std::mt19937 rng_;
    
    // 计算奖励
    double compute_reward(const EpochPerf& curr, const EpochPerf& prev) const {
        return curr.H_cache - prev.H_cache;
    }
    
    // 选择动作
    Action select_action();
    
    // 应用动作，返回新alpha
    double apply_action(Action action);
    
    // 衰减探索率
    void decay_epsilon();
};

// 漂移检测器
class DriftDetector {
public:
    explicit DriftDetector(double threshold = 0.15, int window_size = 3);
    
    // 添加新的观测值，返回是否检测到漂移
    bool observe(double read_ratio);
    
    // 重置
    void reset();
    
    // 获取当前平均读比例
    double current_avg() const;

private:
    double threshold_;
    int window_size_;
    std::deque<double> recent_ratios_;
    double baseline_avg_;
    bool baseline_set_;
};

} // namespace tmpdb

#endif // TMPDB_RL_TUNER_HPP
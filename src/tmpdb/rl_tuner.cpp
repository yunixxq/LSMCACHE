#include "tmpdb/rl_tuner.hpp"
#include <sstream>
#include <iomanip>
#include <numeric>

namespace tmpdb {

std::string action_to_string(Action action) {
    switch (action) {
        case Action::DECREASE_LARGE: return "↓↓";
        case Action::DECREASE_SMALL: return "↓";
        case Action::HOLD:           return "—";
        case Action::INCREASE_SMALL: return "↑";
        case Action::INCREASE_LARGE: return "↑↑";
        default:                     return "?";
    }
}

// 根据动作计算alpha的变化量，将离散动作映射到连续的alpha空间
double action_to_delta(Action action, double step_size) {
    switch (action) {
        case Action::DECREASE_LARGE: return -2.0 * step_size;
        case Action::DECREASE_SMALL: return -step_size;
        case Action::HOLD:           return 0.0;
        case Action::INCREASE_SMALL: return step_size;
        case Action::INCREASE_LARGE: return 2.0 * step_size;
        default:                     return 0.0;
    }
}

RLState::RLState() 
    : read_ratio(0.5), current_alpha(0.5),
      H_cache_prev(0.5), H_cache_delta(0.0),
      latency_prev(100.0), io_prev(1.0),
      episode_count(0) {}

// 两种情况下调用重置：
// - (1)系统启动时用GBDT预测的alpha初始化
// - (2)检测到workload漂移时用新的预测值重置
void RLState::reset(double alpha, double rr) {
    current_alpha = alpha;
    read_ratio = rr;
    H_cache_prev = 0.5;
    H_cache_delta = 0.0;
    latency_prev = 100.0;
    io_prev = 1.0;
    episode_count = 0;
}

// 使用新的性能更新状态，重点是H_cache_delta的计算，
// 这个趋势信息对agent判断调整方向非常重要。
void RLState::update_from_stats(double H_cache, double latency_ms, double io_kb_per_op,
                                 double prev_H_cache) {
    H_cache_delta = H_cache - prev_H_cache;
    H_cache_prev = H_cache;
    latency_prev = latency_ms;
    io_prev = io_kb_per_op;
    episode_count++;
}

QFunction::QFunction() : weights_(FEATURE_DIM, 0.0) {}

std::vector<double> QFunction::extract_features(const RLState& state, Action action) const {
    std::vector<double> phi(FEATURE_DIM, 0.0);
    int action_idx = static_cast<int>(action);
    int offset = action_idx * STATE_FEATURES;
    
    // 状态特征 (针对每个动作有一组)
    phi[offset + 0] = state.read_ratio;
    phi[offset + 1] = state.current_alpha;
    phi[offset + 2] = state.H_cache_prev;
    phi[offset + 3] = state.H_cache_delta;
    phi[offset + 4] = state.latency_prev / 1000.0;  // 归一化到秒
    phi[offset + 5] = state.io_prev / 10.0;         // 归一化
    
    // 动作bias
    phi[STATE_FEATURES * NUM_ACTIONS + action_idx] = 1.0;
    
    return phi;
}

double QFunction::compute(const RLState& state, Action action) const {
    auto phi = extract_features(state, action);
    double q = 0.0;
    for (int i = 0; i < FEATURE_DIM; i++) {
        q += weights_[i] * phi[i];
    }
    return q;
}

Action QFunction::best_action(const RLState& state) const {
    Action best = Action::HOLD;
    double best_q = -1e9;
    for (int a = 0; a < NUM_ACTIONS; a++) {
        Action action = static_cast<Action>(a);
        double q = compute(state, action);
        if (q > best_q) {
            best_q = q;
            best = action;
        }
    }
    return best;
}

void QFunction::update(const RLState& prev_state, Action action, double reward,
                       const RLState& curr_state, double learning_rate, double discount_factor) {
    double q_current = compute(prev_state, action);
    
    // 找下一状态的最大Q值
    double q_next_max = -1e9;
    for (int a = 0; a < NUM_ACTIONS; a++) {
        double q = compute(curr_state, static_cast<Action>(a));
        q_next_max = std::max(q_next_max, q);
    }
    
    // TD误差
    double td_error = reward + discount_factor * q_next_max - q_current;
    
    // 梯度更新
    auto phi = extract_features(prev_state, action);
    for (int i = 0; i < FEATURE_DIM; i++) {
        weights_[i] += learning_rate * td_error * phi[i];
    }
}

std::vector<double> QFunction::get_all_q_values(const RLState& state) const {
    std::vector<double> q_values(NUM_ACTIONS);
    for (int a = 0; a < NUM_ACTIONS; a++) {
        q_values[a] = compute(state, static_cast<Action>(a));
    }
    return q_values;
}

void QFunction::reset() {
    std::fill(weights_.begin(), weights_.end(), 0.0);
}

UCBTracker::UCBTracker(double exploration_weight)
    : action_counts_(NUM_ACTIONS, 0),
      action_rewards_(NUM_ACTIONS, 0.0),
      total_count_(0),
      exploration_weight_(exploration_weight) {}

// Upper Confidence Bound
// 计算UCB值 UCB会选择历史表现好 + 尝试次数少的设置为高UCB值
double UCBTracker::ucb_value(Action action) const {
    int a = static_cast<int>(action);
    // case1.从未尝试过的动作 - 最高优先级
    if (action_counts_[a] == 0) {
        return 1e9;
    }

    // case2. 已探索的动作 exploration_weight_ = 1.414
    double avg = action_rewards_[a] / action_counts_[a]; // 历史平均奖励
    double exploration_bonus = exploration_weight_ * 
        std::sqrt(std::log(static_cast<double>(total_count_)) / action_counts_[a]); // 探索奖励
    return avg + exploration_bonus;
}

void UCBTracker::update(Action action, double reward) {
    int a = static_cast<int>(action);
    action_counts_[a]++;
    action_rewards_[a] += reward;
    total_count_++;
}

void UCBTracker::reset() {
    std::fill(action_counts_.begin(), action_counts_.end(), 0);
    std::fill(action_rewards_.begin(), action_rewards_.end(), 0.0);
    total_count_ = 0;
}

double UCBTracker::avg_reward(Action action) const {
    int a = static_cast<int>(action);
    if (action_counts_[a] == 0) return 0.0;
    return action_rewards_[a] / action_counts_[a];
}

RLTuner::RLTuner(const RLConfig& config)
    : config_(config),
      ucb_tracker_(config.ucb_c),
      epsilon_(config.epsilon_start),
      has_prev_perf_(false),
      last_action_(Action::HOLD),
      last_reward_(0.0),
      epoch_count_(0) {
    rng_.seed(std::random_device{}());
}

void RLTuner::init(double initial_alpha, double initial_read_ratio) {
    state_.reset(initial_alpha, initial_read_ratio);
    q_function_.reset();
    ucb_tracker_.reset();
    epsilon_ = config_.epsilon_start;
    has_prev_perf_ = false;
    last_action_ = Action::HOLD;
    last_reward_ = 0.0;
    epoch_count_ = 0;
}

void RLTuner::on_drift_detected(double new_predicted_alpha, double new_read_ratio) {
    // 注意: Q函数权重保留，实现迁移学习
    state_.reset(new_predicted_alpha, new_read_ratio);
    ucb_tracker_.reset();
    epsilon_ = config_.epsilon_start;  // 重置探索率
    has_prev_perf_ = false;
    epoch_count_ = 0;
    recent_rewards_.clear();
    converged_ = false; 
}

double RLTuner::on_epoch_end(const EpochPerf& perf) {
    epoch_count_++;
    
    // 更新状态中的读比例
    state_.read_ratio = perf.read_ratio;
    
    // 第一个epoch，单独进行记录
    if (!has_prev_perf_) {
        prev_perf_ = perf;
        has_prev_perf_ = true;

        // 初始化状态
        state_.update_from_stats(perf.H_cache, perf.latency_ms, 
                                  perf.io_kb_per_op, perf.H_cache);
        
        // return state_.current_alpha;  // 保持不变-这样的话第二个epoch依然是alpha，实际上是无效的reward
        // 由于现在没有reward，因此使用UCB肢解选择第一个探索动作
        Action first_action = select_action();
        last_action_ = first_action;
        double new_alpha = apply_action(first_action);

        return new_alpha;
    }
    
    // 正常RL更新流程
    
    // 1. 计算奖励
    double reward = compute_reward(perf, prev_perf_);
    last_reward_ = reward;

    // 2. 收敛检测
    recent_rewards_.push_back(std::abs(reward));
    if (recent_rewards_.size() > static_cast<size_t>(convergence_window_)) {
        recent_rewards_.pop_front();
    }

    // 检查是否收敛：最近N个epoch的reward绝对值都小于阈值
    if (recent_rewards_.size() >= static_cast<size_t>(convergence_window_)) {
        bool all_below_threshold = true;
        for (double r : recent_rewards_) {
            if (r > config_.convergence_threshold) {
                all_below_threshold = false;
                break;
            }
        }
        if (all_below_threshold && !converged_) {
            converged_ = true;
        }
    }

    // 3. 更新UCB统计
    ucb_tracker_.update(last_action_, reward);
    
    // 4. 构建当前状态
    RLState prev_state = state_;
    state_.update_from_stats(perf.H_cache, perf.latency_ms, 
                              perf.io_kb_per_op, prev_perf_.H_cache);
    
    // 5. 更新Q函数
    q_function_.update(prev_state, last_action_, reward, state_,
                       config_.learning_rate, config_.discount_factor);
    
    // 6. 选择新动作
    Action action = select_action();
    last_action_ = action;
    
    // 7. 应用动作
    double new_alpha = apply_action(action);
    
    // 8. 衰减探索率
    decay_epsilon();
    
    // 9. 保存当前性能
    prev_perf_ = perf;
    
    return new_alpha;
}

// 结合UCB和ε-greedy，平衡了探索与利用。
Action RLTuner::select_action() {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    // 初始探索期：UCB优先选择未尝试过的动作 
    if (ucb_tracker_.total_count() < NUM_ACTIONS * 2) {
        Action best = Action::HOLD;
        double best_ucb = -1e9;
        for (int a = 0; a < NUM_ACTIONS; a++) {
            Action action = static_cast<Action>(a);
            double u = ucb_tracker_.ucb_value(action);
            if (u > best_ucb) {
                best_ucb = u;
                best = action;
            }
        }
        return best;
    }
    
    // ε-greedy
    if (dist(rng_) < epsilon_) {
        // 探索：基于UCB选择
        Action best = Action::HOLD;
        double best_ucb = -1e9;
        for (int a = 0; a < NUM_ACTIONS; a++) {
            Action action = static_cast<Action>(a);
            double u = ucb_tracker_.ucb_value(action);
            if (u > best_ucb) {
                best_ucb = u;
                best = action;
            }
        }
        return best;
    } else {
        // 利用：基于Q值选择
        return q_function_.best_action(state_);
    }
}

double RLTuner::apply_action(Action action) {
    double delta = action_to_delta(action, config_.step_size);
    double new_alpha = state_.current_alpha + delta;
    
    // Clip到有效范围
    new_alpha = std::max(config_.alpha_min, std::min(config_.alpha_max, new_alpha));
    
    state_.current_alpha = new_alpha;
    return new_alpha;
}

void RLTuner::decay_epsilon() {
    epsilon_ = std::max(config_.epsilon_min, epsilon_ * config_.epsilon_decay);
}

void RLTuner::set_seed(int seed) {
    rng_.seed(seed);
}

std::string RLTuner::get_stats_string() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3)
        << "ε=" << epsilon_
        << ", action=" << action_to_string(last_action_)
        << ", reward=" << last_reward_;
    return oss.str();
}

std::vector<double> RLTuner::get_q_values() const {
    return q_function_.get_all_q_values(state_);
}

DriftDetector::DriftDetector(double threshold, int window_size)
    : threshold_(threshold),
      window_size_(window_size),
      baseline_avg_(0.0),
      baseline_set_(false) {}

bool DriftDetector::observe(double read_ratio) {
    recent_ratios_.push_back(read_ratio);
    
    // 保持窗口大小
    while (static_cast<int>(recent_ratios_.size()) > window_size_) {
        recent_ratios_.pop_front();
    }
    
    // 窗口未满，不检测
    if (static_cast<int>(recent_ratios_.size()) < window_size_) {
        return false;
    }
    
    // 计算当前窗口平均
    double curr_avg = current_avg();
    
    // 设置baseline
    if (!baseline_set_) {
        baseline_avg_ = curr_avg;
        baseline_set_ = true;
        return false;
    }
    
    // 检测漂移
    if (std::abs(curr_avg - baseline_avg_) > threshold_) {
        // 检测到漂移，重置baseline
        baseline_avg_ = curr_avg;
        return true;
    }
    
    // 缓慢更新baseline（指数移动平均）
    baseline_avg_ = 0.9 * baseline_avg_ + 0.1 * curr_avg;
    
    return false;
}

void DriftDetector::reset() {
    recent_ratios_.clear();
    baseline_avg_ = 0.0;
    baseline_set_ = false;
}

double DriftDetector::current_avg() const {
    if (recent_ratios_.empty()) return 0.0;
    double sum = std::accumulate(recent_ratios_.begin(), recent_ratios_.end(), 0.0);
    return sum / recent_ratios_.size();
}

} // namespace tmpdb
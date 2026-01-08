#pragma once

#include <chrono>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>

class ProgressBar {
public:
    ProgressBar(size_t total, const std::string& stage_name, size_t bar_width = 40)
        : total_(total)
        , current_(0)
        , bar_width_(bar_width)
        , stage_name_(stage_name)
        , start_time_(std::chrono::steady_clock::now())
        , last_update_time_(start_time_)
        , update_interval_ms_(100)  // 每100ms更新一次显示
    {
        display();
    }

    void update(size_t count = 1) {
        current_ += count;
        
        auto now = std::chrono::steady_clock::now();
        auto elapsed_since_update = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - last_update_time_).count();
        
        // 限制更新频率，避免频繁刷新
        if (elapsed_since_update >= update_interval_ms_ || current_ >= total_) {
            display();
            last_update_time_ = now;
        }
    }

    void set_current(size_t current) {
        current_ = current;
        auto now = std::chrono::steady_clock::now();
        auto elapsed_since_update = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - last_update_time_).count();
        
        if (elapsed_since_update >= update_interval_ms_ || current_ >= total_) {
            display();
            last_update_time_ = now;
        }
    }

    void finish() {
        current_ = total_;
        display();
        std::cout << std::endl;
    }

    ~ProgressBar() {
        // 确保进度条完成时换行
        if (current_ < total_) {
            std::cout << std::endl;
        }
    }

private:
    void display() {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - start_time_).count();

        double progress = static_cast<double>(current_) / static_cast<double>(total_);
        size_t filled = static_cast<size_t>(progress * bar_width_);

        // 计算预计剩余时间
        std::string eta_str;
        if (current_ > 0 && current_ < total_) {
            double rate = static_cast<double>(current_) / (elapsed / 1000.0);
            double remaining_sec = (total_ - current_) / rate;
            eta_str = format_time(static_cast<int64_t>(remaining_sec * 1000));
        } else if (current_ >= total_) {
            eta_str = "00:00";
        } else {
            eta_str = "--:--";
        }

        // 构建进度条
        std::ostringstream oss;
        oss << "\r" << stage_name_ << " |";
        
        for (size_t i = 0; i < bar_width_; ++i) {
            if (i < filled) {
                oss << "█";
            } else if (i == filled) {
                oss << "▓";
            } else {
                oss << "░";
            }
        }
        
        oss << "| " << std::setw(6) << std::fixed << std::setprecision(1) 
            << (progress * 100.0) << "% "
            << "[" << format_number(current_) << "/" << format_number(total_) << "] "
            << "Elapsed: " << format_time(elapsed) << " "
            << "ETA: " << eta_str << "   ";  // 额外空格清除残留字符

        std::cout << oss.str() << std::flush;
    }

    std::string format_time(int64_t ms) const {
        int64_t total_seconds = ms / 1000;
        int64_t hours = total_seconds / 3600;
        int64_t minutes = (total_seconds % 3600) / 60;
        int64_t seconds = total_seconds % 60;

        std::ostringstream oss;
        if (hours > 0) {
            oss << hours << ":" 
                << std::setw(2) << std::setfill('0') << minutes << ":"
                << std::setw(2) << std::setfill('0') << seconds;
        } else {
            oss << std::setw(2) << std::setfill('0') << minutes << ":"
                << std::setw(2) << std::setfill('0') << seconds;
        }
        return oss.str();
    }

    std::string format_number(size_t num) const {
        if (num >= 1000000) {
            return std::to_string(num / 1000000) + "." + 
                   std::to_string((num % 1000000) / 100000) + "M";
        } else if (num >= 1000) {
            return std::to_string(num / 1000) + "." + 
                   std::to_string((num % 1000) / 100) + "K";
        }
        return std::to_string(num);
    }

    size_t total_;
    size_t current_;
    size_t bar_width_;
    std::string stage_name_;
    std::chrono::steady_clock::time_point start_time_;
    std::chrono::steady_clock::time_point last_update_time_;
    int64_t update_interval_ms_;
};
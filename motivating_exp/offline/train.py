import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import logging
import pickle
import sys
import time
import os
import yaml
import warnings

from sklearn.inspection import permutation_importance

warnings.filterwarnings('ignore', message='X does not have valid feature names')

sys.path.append("./motivating_exp")
from lsm_tree.lsm import build_features

np.set_printoptions(suppress=True)

# 基础特征
BASE_FEATURE_NAMES = [
    'read_write_ratio', 'skewness', 'T',
    'M_MB', 'N_over_M', 'alpha', 
    # 'Mbuf_MB', 'Mcache_MB',
]

# 扩展特征
EXTEND_FEATURE_NAMES = [
    'L', 'Wamp',
    # 'flush_rate', 'compaction_rate', 'sst_inv_rate', 'cache_inv_rate'
]

# 完整特征
FULL_FEATURE_NAMES = BASE_FEATURE_NAMES + EXTEND_FEATURE_NAMES

class ModelEvaluator:
    def __init__(self, config_path: str = "motivating_exp/config/config_sampling_exp.yaml"):
        """加载配置"""
        with open(config_path) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        
        # 设置日志
        self.logger = logging.getLogger("cost_model_trainer")
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s', 
                                        datefmt='%H:%M:%S')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.random_state = 42
        self.models = {}
    
    def load_data(self) -> pd.DataFrame:
        """加载采样数据"""
        data_path = "data/sampling_exp_results.csv"
        # self.config["output_path"]["sampling_exp_output"]
        self.logger.info(f"Loading data from: {data_path}")
        samples = pd.read_csv(data_path)
        samples = samples.sample(frac=1, random_state=self.random_state)
        
        self.logger.info(f"Total samples loaded: {len(samples)}")
        return samples
    

    
    def prepare_data(self, samples: pd.DataFrame, feature_type: str = "full"):
        """准备训练数据"""
        X = []
        Y = []
        
        for _, sample in samples.iterrows():
            X.append(build_features(
                sample['N'], sample['M_MB'], sample['T'], sample['alpha'],
                sample['read_ratio'], sample['write_ratio'], sample['skewness'],
                feature_type
            ))

            Y.append(sample["H_cache"])
        
        X = np.array(X)
        Y = np.array(Y)
        
        feature_names = FULL_FEATURE_NAMES if feature_type == "full" else BASE_FEATURE_NAMES
        return X, Y, feature_names
    
    def train_model(self, X: np.ndarray, Y: np.ndarray, 
                        model_type: str = "xgboost", feature_type: str = "full"):
        """训练模型"""
        model_name = f"{model_type}_{feature_type}"
        self.logger.info(f"Training {model_name}...")
        
        start_time = time.time()
        
        if model_type == "xgboost":
            model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=self.random_state
            )
        else:
            model = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                num_leaves=31,
                random_state=self.random_state,
                verbose=-1
            )
        
        model.fit(X, Y)
        
        elapsed = time.time() - start_time
        self.logger.info(f"  Training time: {elapsed:.2f}s")
        
        self.models[model_name] = model
        return model
   
    def analyze_gain_importance(self, model, feature_names: list, 
                                 model_type: str = "xgboost") -> pd.DataFrame:
        """
        方法1: Gain-based 特征重要性
        
        原理: 统计每个特征在分裂时带来的信息增益总和
        - 增益越高，说明该特征对降低预测误差贡献越大
        - 模型内置计算，无需额外数据
        """
        self.logger.info("\n[Method 1: Gain-based Importance]")
        self.logger.info("原理: 统计特征在树分裂时带来的信息增益总和")
        self.logger.info("-" * 50)
        
        if model_type == "lightgbm":
            importance = model.booster_.feature_importance(importance_type='gain')
        else:
            importance = model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'gain': importance
        }).sort_values('gain', ascending=False)
        
        importance_df['gain_pct'] = importance_df['gain'] / importance_df['gain'].sum() * 100
        
        for rank, (_, row) in enumerate(importance_df.iterrows(), 1):
            marker = "★" if row['feature'] in EXTEND_FEATURE_NAMES else " "
            self.logger.info(f"  {rank:2d}. {marker} {row['feature']:20s}: {row['gain_pct']:6.2f}%")
        
        return importance_df

    def analyze_permutation_importance(self, model, X: np.ndarray, Y: np.ndarray,
                                        feature_names: list, n_repeats: int = 10) -> pd.DataFrame:
        """
        方法2: Permutation 特征重要性
        
        原理: 打乱某特征的值，观察模型性能下降程度
        - 性能下降越多，说明模型越依赖该特征

        """
        self.logger.info("\n[Method 2: Permutation Importance]")
        self.logger.info("原理: 打乱特征值后观察预测性能下降程度")
        self.logger.info(f"参数: n_repeats={n_repeats} (重复次数，用于计算标准差)")
        self.logger.info("-" * 50)
        
        
        # 直接在全量数据上计算（不划分）
        perm_result = permutation_importance(
            model, X, Y, 
            n_repeats=n_repeats, 
            random_state=self.random_state,
            scoring='neg_mean_absolute_error'
        )
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'perm_mean': perm_result.importances_mean,
            'perm_std': perm_result.importances_std
        }).sort_values('perm_mean', ascending=False)
        
        # 归一化
        total = importance_df['perm_mean'].sum()
        importance_df['perm_pct'] = importance_df['perm_mean'] / total * 100 if total > 0 else 0
        
        for rank, (_, row) in enumerate(importance_df.iterrows(), 1):
            marker = "★" if row['feature'] in EXTEND_FEATURE_NAMES else " "
            self.logger.info(f"  {rank:2d}. {marker} {row['feature']:20s}: {row['perm_pct']:6.2f}% (±{row['perm_std']:.4f})")
        
        return importance_df
    
    def analyze_feature_importance(self, model, X: np.ndarray, Y: np.ndarray,
                                    feature_names: list, model_name: str) -> pd.DataFrame:
        """
        综合特征重要性分析（只对 Full 模型）
        使用两种方法相互验证
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info(f"Feature Importance Analysis: {model_name}")
        self.logger.info("=" * 60)
        self.logger.info("★ 标记表示五阶段耦合指标")
        
        model_type = "lightgbm" if "lightgbm" in model_name else "xgboost"
        
        # 方法1: Gain-based
        gain_df = self.analyze_gain_importance(model, feature_names, model_type)
        
        # 方法2: Permutation
        perm_df = self.analyze_permutation_importance(model, X, Y, feature_names)
        
        # 合并结果
        combined_df = gain_df.merge(
            perm_df[['feature', 'perm_mean', 'perm_std', 'perm_pct']], 
            on='feature'
        )
        
        # 标记五阶段指标
        combined_df['is_coupling'] = combined_df['feature'].isin(EXTEND_FEATURE_NAMES)
        
        # 统计五阶段指标总贡献
        coupling_gain = combined_df[combined_df['is_coupling']]['gain_pct'].sum()
        coupling_perm = combined_df[combined_df['is_coupling']]['perm_pct'].sum()
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Summary: Five-Stage Coupling Features Contribution")
        self.logger.info("=" * 60)
        self.logger.info(f"  Gain-based total:       {coupling_gain:6.2f}%")
        self.logger.info(f"  Permutation total:      {coupling_perm:6.2f}%")
        self.logger.info("")
        self.logger.info("Per-feature breakdown:")
        for feature in EXTEND_FEATURE_NAMES:
            row = combined_df[combined_df['feature'] == feature].iloc[0]
            self.logger.info(f"  - {feature:20s}: Gain={row['gain_pct']:5.2f}%, Perm={row['perm_pct']:5.2f}%")
        
        return combined_df
    
    def save_models(self):
        """保存所有模型"""
        for model_name, model in self.models.items():
            # 根据模型名称获取对应的配置路径
            if model_name == "xgboost_base":
                model_path = self.config["xgb_model"]["calm_xgb_base_model"]
            elif model_name == "xgboost_full":
                model_path = self.config["xgb_model"]["calm_xgb_full_model"]
            elif model_name == "lightgbm_base":
                model_path = self.config["lgb_model"]["calm_lgb_base_model"]
            elif model_name == "lightgbm_full":
                model_path = self.config["lgb_model"]["calm_lgb_full_model"]
            else:
                # 默认路径
                model_path = f"models/calm_{model_name}.pkl"
            
            # 确保目录存在
            dir_name = os.path.dirname(model_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            self.logger.info(f"Model saved: {model_path}")

    def save_feature_importance(self, importance_results: dict):
        """保存特征重要性结果"""
        output_dir = self.config.get("output_path", {}).get("results_dir", "results")
        os.makedirs(output_dir, exist_ok=True)
        
        for model_name, df in importance_results.items():
            output_path = os.path.join(output_dir, f"feature_importance_{model_name}.csv")
            df.to_csv(output_path, index=False)
            self.logger.info(f"Feature importance saved: {output_path}")
    
    def run(self):
        """运行完整训练流程"""
        self.logger.info("=" * 60)
        self.logger.info("CALM Cost Model Training")
        self.logger.info("=" * 60)
        
        # 1. 加载数据
        samples = self.load_data()
        
        importance_results = {}
        
        # 2. 训练 Base 模型（不做特征重要性分析）
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Training BASE Models (No Feature Importance Analysis)")
        self.logger.info("=" * 60)
        
        X_base, Y_base, _ = self.prepare_data(samples, "base")
        self.logger.info(f"Base feature matrix shape: {X_base.shape}")
        
        self.train_model(X_base, Y_base, "xgboost", "base")
        self.train_model(X_base, Y_base, "lightgbm", "base")
        
        # 3. 训练 Full 模型（做特征重要性分析）
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Training FULL Models (With Feature Importance Analysis)")
        self.logger.info("=" * 60)
        
        X_full, Y_full, feature_names = self.prepare_data(samples, "full")
        self.logger.info(f"Full feature matrix shape: {X_full.shape}")
        
        # XGBoost Full
        xgb_model = self.train_model(X_full, Y_full, "xgboost", "full")
        importance_results["xgboost_full"] = self.analyze_feature_importance(
            xgb_model, X_full, Y_full, feature_names, "xgboost_full"
        )
        
        # LightGBM Full
        lgb_model = self.train_model(X_full, Y_full, "lightgbm", "full")
        importance_results["lightgbm_full"] = self.analyze_feature_importance(
            lgb_model, X_full, Y_full, feature_names, "lightgbm_full"
        )
        
        # 4. 汇总对比
        self.print_final_summary(importance_results)
        
        # 5. 保存
        self.save_models()
        self.save_feature_importance(importance_results)
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Training completed!")
        self.logger.info("=" * 60)
        
        return self.models, importance_results

    def print_final_summary(self, importance_results: dict):
        """输出最终汇总"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("FINAL SUMMARY: Five-Stage Coupling Features Value")
        self.logger.info("=" * 60)
        
        self.logger.info("\n两种验证方法说明:")
        self.logger.info("  1. Gain-based:   基于树分裂时的信息增益（模型内置）")
        self.logger.info("  2. Permutation:  打乱特征后的性能下降（更可靠）")
        
        self.logger.info("\n各模型五阶段指标总贡献:")
        self.logger.info("-" * 50)
        self.logger.info(f"{'Model':<20} {'Gain-based':<15} {'Permutation':<15}")
        self.logger.info("-" * 50)
        
        for model_name, df in importance_results.items():
            coupling_gain = df[df['is_coupling']]['gain_pct'].sum()
            coupling_perm = df[df['is_coupling']]['perm_pct'].sum()
            self.logger.info(f"{model_name:<20} {coupling_gain:>10.2f}%     {coupling_perm:>10.2f}%")
        
        self.logger.info("-" * 50)
        self.logger.info("\n结论: 如果五阶段指标贡献占比显著 (>30%)，")
        self.logger.info("      说明理论指导的特征工程是有价值的。")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "motivating_exp/config/config_sampling_exp.yaml"
    
    trainer = ModelEvaluator(config_path)
    models, importance_results = trainer.run()
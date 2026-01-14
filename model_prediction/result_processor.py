import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List

class ResultProcessor:
    def __init__(self, params: Dict[str, Any] = None):
        self.params = params or {}
        self.results_dir = self.params.get('results_dir', 'model_prediction/results')
        os.makedirs(self.results_dir, exist_ok=True)
    
    def process_prediction_result(self, prediction_result: Dict[str, Any], original_data: pd.DataFrame) -> Dict[str, Any]:
        """处理预测结果，计算统计信息和与实际数据的比较"""
        predictions = prediction_result['predictions']
        model_info = prediction_result['model_info']
        
        result = {
            'predictions': predictions,
            'model_info': model_info,
            'stats': {
                'mean_prediction': float(np.mean(predictions)),
                'std_prediction': float(np.std(predictions)),
                'min_prediction': float(np.min(predictions)),
                'max_prediction': float(np.max(predictions))
            },
            'comparison': self._compare_with_actual(original_data, predictions),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        return result
    
    def _compare_with_actual(self, original_data: pd.DataFrame, predictions: List[float]) -> Dict[str, Any]:
        """比较预测与实际数据，计算误差指标"""
        if 'close' in original_data.columns and len(predictions) > 0:
            actual = original_data['close'].values
            if len(actual) > len(predictions):
                actual = actual[:len(predictions)]
            elif len(actual) < len(predictions):
                predictions = predictions[:len(actual)]
            
            mse = float(np.mean((np.array(actual) - np.array(predictions)) ** 2))
            rmse = float(np.sqrt(mse))
            mae = float(np.mean(np.abs(np.array(actual) - np.array(predictions))))
            
            return {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'actual_values': actual.tolist()[:10],
                'predicted_values': predictions[:10]
            }
        return {}
    
    def save_result(self, result: Dict[str, Any], filename: str = 'prediction_result.json') -> str:
        """保存结果到JSON文件"""
        output_path = os.path.join(self.results_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        return output_path
    
    def load_result(self, filename: str = 'prediction_result.json') -> Optional[Dict[str, Any]]:
        """从JSON文件加载结果"""
        input_path = os.path.join(self.results_dir, filename)
        if not os.path.exists(input_path):
            return None
        
        with open(input_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_performance_report(self, model_info: Dict[str, Any], prediction_stats: Dict[str, Any]) -> str:
        """保存性能报告"""
        report = {
            'model_info': model_info,
            'prediction_stats': prediction_stats,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        return self.save_result(report, 'performance_report.json')
    
    def get_result_files(self) -> List[Dict[str, Any]]:
        """获取结果文件列表"""
        result_files = []
        
        if not os.path.exists(self.results_dir):
            return result_files
        
        for filename in os.listdir(self.results_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(self.results_dir, filename)
                file_stat = os.stat(file_path)
                
                result_files.append({
                    'filename': filename,
                    'size': file_stat.st_size,
                    'modified_at': os.path.getmtime(file_path)
                })
        
        return result_files

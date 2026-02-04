import torch
import torch.nn as nn
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class QuantileTransformerRegressor(nn.Module):
    """Quantile Regression을 위한 Transformer 기반 모델"""
    
    def __init__(self, input_dim, num_quantiles=999, d_model=128, nhead=8, 
                 num_layers=3, dim_feedforward=512, dropout=0.1):
        super(QuantileTransformerRegressor, self).__init__()
        self.num_quantiles = num_quantiles
        
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 1, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, dim_feedforward // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 4, num_quantiles)
        )
        
    def forward(self, x):
        x = self.input_embedding(x)
        x = x.unsqueeze(1) + self.pos_encoder
        x = self.transformer_encoder(x)
        return self.fc_out(x.squeeze(1))


class ProbabilityPredictor:
    """TFT 4-Feature 모델을 사용한 확률 예측 클래스"""
    
    def __init__(self, model_path='./results_tft_4feat/best_model.pt'):
        self.model_path = model_path
        self.device = device
        self.quantiles = np.linspace(0.001, 0.999, 999)
        self.feature_names = ['예가범위', '낙찰하한율', '추정가격', '기초금액']
        self.model = self._load_model()
        self.scaler = None
        
    def _load_model(self):
        """학습된 모델 로드"""
        model = QuantileTransformerRegressor(
            input_dim=4, num_quantiles=999, d_model=128, nhead=8,
            num_layers=3, dim_feedforward=512, dropout=0.1
        ).to(self.device)
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.eval()
        
        print(f"✓ 모델 로드 완료: {self.model_path}")
        if 'epoch' in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}, Val Loss: {checkpoint.get('val_loss', 0):.6f}")
        
        return model
    
    def _prepare_input(self, input_features):
        """입력 피처를 numpy array로 변환"""
        if isinstance(input_features, dict):
            X = np.array([[
                input_features['예가범위'],
                input_features['낙찰하한율'],
                input_features['추정가격'],
                input_features['기초금액']
            ]], dtype=np.float32)
        else:
            X = np.array([input_features], dtype=np.float32)
            if X.shape[1] != 4:
                raise ValueError(f"입력 피처는 4개여야 합니다. 현재: {X.shape[1]}개")
        
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        return X
    
    def _predict_quantiles(self, X):
        """999개 quantile 예측"""
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            return self.model(X_tensor).cpu().numpy()[0]
    
    def _get_input_features_dict(self, X):
        """입력 피처를 dict 형태로 반환"""
        return {
            '예가범위': float(X[0, 0]),
            '낙찰하한율': float(X[0, 1]),
            '추정가격': float(X[0, 2]),
            '기초금액': float(X[0, 3])
        }
    
    def predict_probability(self, input_features, lower_bound, upper_bound):
        """특정 구간의 확률 예측"""
        X = self._prepare_input(input_features)
        pred_quantiles = self._predict_quantiles(X)
        
        # 구간 내 확률 계산
        lower_idx = np.searchsorted(pred_quantiles, lower_bound, side='left')
        upper_idx = np.searchsorted(pred_quantiles, upper_bound, side='right')
        probability = (upper_idx - lower_idx) / len(pred_quantiles)
        
        return {
            'probability': float(probability),
            'probability_percent': float(probability * 100),
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'lower_quantile_index': int(lower_idx),
            'upper_quantile_index': int(upper_idx),
            'median_prediction': float(pred_quantiles[499]),
            'mean_prediction': float(np.mean(pred_quantiles)),
            'input_features': self._get_input_features_dict(X)
        }
    
    def get_prediction_intervals(self, input_features, confidence_levels=[0.5, 0.8, 0.9, 0.95]):
        """여러 신뢰구간 예측"""
        X = self._prepare_input(input_features)
        pred_quantiles = self._predict_quantiles(X)
        
        intervals = {}
        for conf in confidence_levels:
            lower_idx = int((1 - conf) / 2 * 999)
            upper_idx = int((1 + conf) / 2 * 999)
            
            intervals[f'{int(conf*100)}%'] = {
                'lower': float(pred_quantiles[lower_idx]),
                'upper': float(pred_quantiles[upper_idx]),
                'median': float(pred_quantiles[499]),
                'width': float(pred_quantiles[upper_idx] - pred_quantiles[lower_idx])
            }
        
        return {
            'intervals': intervals,
            'median_prediction': float(pred_quantiles[499]),
            'mean_prediction': float(np.mean(pred_quantiles)),
            'input_features': self._get_input_features_dict(X)
        }
    
    
    def get_highest_probability_ranges(self, input_features, bin_width=0.001, top_k=3):
        """
        Quantile Function을 PDF로 변환하여 확률 밀도가 높은 구간 찾기
        
        수학적 원리:
        - Quantile Function: Q(τ) = y, τ ∈ [0.001, 0.999]
        - CDF: F(y) = τ (역함수 관계)
        - PDF: f(y) = dF(y)/dy = dτ/dy
        
        이산 근사:
        - f(y_i) ≈ Δτ / ΔQ = (τ_{i+1} - τ_{i-1}) / (Q_{i+1} - Q_{i-1})
        """
        X = self._prepare_input(input_features)
        pred_quantiles = self._predict_quantiles(X)  # Q(τ_i) for i=0..998
        
        # 🔍 단조성 검사
        non_monotonic = np.diff(pred_quantiles) < 0
        if np.any(non_monotonic):
            n_violations = np.sum(non_monotonic)
            print(f"⚠️  경고: Quantile Function이 {n_violations}개 구간에서 감소합니다!")
            print(f"   이는 역함수가 정의되지 않는 구간입니다.")
            violation_indices = np.where(non_monotonic)[0][:5]  # 처음 5개만
            for idx in violation_indices:
                print(f"   τ={self.quantiles[idx]:.3f}: Q={pred_quantiles[idx]:.4f} → Q={pred_quantiles[idx+1]:.4f}")
        
        # 1. PDF 계산: f(y) = Δτ / ΔQ
        pdf_values = np.zeros(len(pred_quantiles))
        
        # 중심차분으로 PDF 계산 (양 끝 제외)
        for i in range(1, len(pred_quantiles) - 1):
            delta_tau = self.quantiles[i+1] - self.quantiles[i-1]  # 0.002
            delta_Q = pred_quantiles[i+1] - pred_quantiles[i-1]
            
            if abs(delta_Q) > 1e-10:  # 0으로 나누기 방지
                pdf_values[i] = delta_tau / delta_Q
                # 음수 PDF 방지 (비단조 구간)
                if pdf_values[i] < 0:
                    pdf_values[i] = 0  # 음수 확률밀도는 0으로 처리
            else:
                pdf_values[i] = 100.0  # 매우 높은 밀도 (하지만 현실적인 값)
        
        # 양 끝점 처리 (전진/후진 차분)
        if len(pred_quantiles) > 1:
            # 첫 점 (전진차분)
            delta_tau_0 = self.quantiles[1] - self.quantiles[0]
            delta_Q_0 = pred_quantiles[1] - pred_quantiles[0]
            if abs(delta_Q_0) > 1e-10:
                pdf_values[0] = max(0, delta_tau_0 / delta_Q_0)  # 음수 방지
            else:
                pdf_values[0] = 100.0
            
            # 마지막 점 (후진차분)
            delta_tau_last = self.quantiles[-1] - self.quantiles[-2]
            delta_Q_last = pred_quantiles[-1] - pred_quantiles[-2]
            if abs(delta_Q_last) > 1e-10:
                pdf_values[-1] = max(0, delta_tau_last / delta_Q_last)  # 음수 방지
            else:
                pdf_values[-1] = 100.0
        
        # 2. bin_width 단위로 구간을 나누고 평균 PDF 계산
        # min/max를 bin_width 단위로 정렬하여 깔끔한 경계 생성
        min_val = float(pred_quantiles.min())
        max_val = float(pred_quantiles.max())
        
        # bin_width 단위로 내림/올림하여 정밀도 맞춤
        min_aligned = np.floor(min_val / bin_width) * bin_width
        max_aligned = np.ceil(max_val / bin_width) * bin_width
        
        bins = np.arange(min_aligned, max_aligned + bin_width, bin_width)
        
        bin_info = []
        for i in range(len(bins) - 1):
            lower, upper = bins[i], bins[i + 1]
            
            # 이 구간에 속하는 quantile 찾기
            in_bin = (pred_quantiles >= lower) & (pred_quantiles < upper if i < len(bins) - 2 else pred_quantiles <= upper)
            quantile_indices = np.where(in_bin)[0]
            
            if len(quantile_indices) == 0:
                continue
            
            # 구간 내 평균 PDF (확률밀도)
            avg_pdf = float(np.mean(pdf_values[quantile_indices]))
            
            # 구간의 확률 ≈ ∫ f(y) dy ≈ f(y) × Δy
            probability = avg_pdf * bin_width
            
            bin_info.append({
                'range': f'{(lower-1)*100:+.1f}% ~ {(upper-1)*100:+.1f}%',  # 증감으로 표시, 0.1%p 단위, ~ 앞뒤 공백
                'lower': float(lower),
                'upper': float(upper),
                'center': float((lower + upper) / 2),
                'pdf': avg_pdf,  # 확률밀도 f(y)
                'probability': float(probability),  # P(y ∈ [lower, upper]) - 정규화 전
                'probability_percent': float(probability * 100)
            })
        
        # 전체 확률 정규화 (∑P = 1이 되도록)
        total_probability = sum(b['probability'] for b in bin_info)
        print(f"[DEBUG] 정규화 전 total_probability: {total_probability:.4f}")
        
        if total_probability > 0:
            for b in bin_info:
                old_prob = b['probability']
                b['probability'] = b['probability'] / total_probability
                b['probability_percent'] = b['probability'] * 100
                if old_prob > 1.0:  # 100% 초과한 구간만 출력
                    print(f"[DEBUG] 구간 [{b['lower']:.2f}, {b['upper']:.2f}]: {old_prob*100:.2f}% → {b['probability_percent']:.2f}%")
        
        # PDF 기준으로 정렬 (확률밀도가 높은 순)
        sorted_bins = sorted(bin_info, key=lambda x: x['pdf'], reverse=True)
        
        return {
            'top_ranges': sorted_bins[:top_k],
            'all_ranges': sorted_bins,
            'total_bins': len(sorted_bins),
            'bin_width': bin_width,
            'prediction_range': {'min': min_val, 'max': max_val, 'range': max_val - min_val},
            'statistics': {
                'median': float(pred_quantiles[499]),
                'mean': float(np.mean(pred_quantiles)),
                'std': float(np.std(pred_quantiles)),
                'q25': float(pred_quantiles[249]),
                'q75': float(pred_quantiles[749])
            },
            'input_features': self._get_input_features_dict(X)
        }
    
    def get_most_probable_range(self, input_features, bin_width=0.5):
        """가장 확률 밀도가 높은 구간 1개 반환"""
        result = self.get_highest_probability_ranges(input_features, bin_width, top_k=1)
        
        if not result['top_ranges']:
            return None
            
        most_probable = result['top_ranges'][0]
        return {
            'most_probable_range': most_probable['range'],
            'lower': most_probable['lower'],
            'upper': most_probable['upper'],
            'center': most_probable['center'],
            'probability': most_probable['probability'],
            'probability_percent': most_probable['probability_percent'],
            'statistics': result['statistics'],
            'prediction_range': result['prediction_range'],
            'input_features': result['input_features']
        }
    
    def get_mode_and_peak_density(self, input_features, bandwidth=0.001):
        """최빈값(mode)과 peak 밀도 분석"""
        X = self._prepare_input(input_features)
        pred_quantiles = self._predict_quantiles(X)
        
        # 밀도 계산
        densities = np.array([
            np.sum(np.abs(pred_quantiles - q_val) <= bandwidth) / 999 / (2 * bandwidth)
            for q_val in pred_quantiles
        ])
        
        # 최대 밀도 인덱스
        peak_idx = np.argmax(densities)
        mode_value = float(pred_quantiles[peak_idx])
        peak_lower, peak_upper = mode_value - bandwidth, mode_value + bandwidth
        peak_count = np.sum((pred_quantiles >= peak_lower) & (pred_quantiles <= peak_upper))
        
        return {
            'mode': mode_value,
            'mode_quantile': float(self.quantiles[peak_idx]),
            'peak_density': float(densities[peak_idx]),
            'peak_range': {
                'lower': float(peak_lower),
                'upper': float(peak_upper),
                'probability': float(peak_count / 999),
                'probability_percent': float(peak_count / 999 * 100)
            },
            'median': float(pred_quantiles[499]),
            'mean': float(np.mean(pred_quantiles)),
            'std': float(np.std(pred_quantiles)),
            'input_features': self._get_input_features_dict(X)
        }
    
    def evaluate_highest_probability_average(self, test_data_path='../dataset/dataset_feature_selected.csv', bin_width=0.001, max_samples=None):
        """
        테스트 데이터의 모든 샘플에 대해 최대 확률 구간의 평균값으로 예측하고 오차율 계산
        
        Args:
            test_data_path: 테스트 데이터 CSV 파일 경로
            bin_width: 구간 폭 (기본값 0.001 = 0.1%p)
            max_samples: 처리할 최대 샘플 수 (None이면 전체)
        
        Returns:
            dict: 평균 오차율 및 상세 통계
        """
        import pandas as pd
        
        print(f"\n{'='*80}")
        print(f"테스트 데이터 평가: 최대 확률 구간의 평균값으로 예측")
        print(f"{'='*80}\n")
        
        # 데이터 로드
        df = pd.read_csv(test_data_path)
        print(f"✓ 테스트 데이터 로드: {len(df)}개 샘플")
        
        # 샘플 수 제한
        if max_samples is not None and len(df) > max_samples:
            df = df.head(max_samples)
            print(f"  처리할 샘플 수 제한: {max_samples}개")
        
        print(f"  컬럼: {list(df.columns)}")
        print(f"  처리할 샘플 수: {len(df)}개\n")
        
        # 실제 사정율은 데이터셋에 이미 존재하는 '사정율' 컬럼 사용
        if '사정율' not in df.columns:
            raise ValueError("데이터셋에 '사정율' 컬럼이 없습니다.")
        
        results = []
        errors = []
        
        print("샘플별 예측 시작...")
        for idx, row in df.iterrows():
            # 입력 피처 준비
            input_features = {
                '예가범위': row['예가범위'],
                '낙찰하한율': row['낙찰하한율'],
                '추정가격': row['추정가격'],
                '기초금액': row['기초금액']
            }
            
            # 최대 확률 구간 찾기
            result = self.get_highest_probability_ranges(input_features, bin_width=bin_width, top_k=1)
            
            if not result['top_ranges']:
                print(f"  경고: 샘플 {idx}에서 구간을 찾을 수 없습니다.")
                continue
            
            # 최대 확률 구간의 중심값 (평균)
            top_range = result['top_ranges'][0]
            predicted_rate = top_range['center']  # 구간의 평균값
            actual_rate = row['사정율']
            
            # 오차 계산
            error = abs(predicted_rate - actual_rate)
            error_percent = error * 100  # %p 단위
            relative_error = (error / actual_rate) * 100  # 상대 오차 (%)
            
            results.append({
                'index': idx,
                'actual_rate': actual_rate,
                'predicted_rate': predicted_rate,
                'error': error,
                'error_percent': error_percent,
                'relative_error': relative_error,
                'probability': top_range['probability_percent'],
                'range_lower': top_range['lower'],
                'range_upper': top_range['upper']
            })
            
            errors.append(error)
            
            # 100개마다 진행상황 출력
            if (idx + 1) % 100 == 0:
                print(f"  진행: {idx + 1}/{len(df)} 샘플 처리 완료...")
        
        # 통계 계산
        errors = np.array(errors)
        error_percents = errors * 100
        
        statistics = {
            'total_samples': len(results),
            'mean_absolute_error': float(np.mean(errors)),
            'mean_absolute_error_percent': float(np.mean(error_percents)),
            'median_absolute_error': float(np.median(errors)),
            'median_absolute_error_percent': float(np.median(error_percents)),
            'std_error': float(np.std(errors)),
            'std_error_percent': float(np.std(error_percents)),
            'min_error': float(np.min(errors)),
            'max_error': float(np.max(errors)),
            'q25_error': float(np.percentile(errors, 25)),
            'q75_error': float(np.percentile(errors, 75)),
            'mean_relative_error_percent': float(np.mean([r['relative_error'] for r in results]))
        }
        
        # 결과 출력
        print(f"\n{'='*80}")
        print(f"평가 결과")
        print(f"{'='*80}")
        print(f"총 샘플 수: {statistics['total_samples']}")
        print(f"\n[절대 오차 (사정율 차이)]")
        print(f"  평균 오차율: {statistics['mean_absolute_error']:.6f} ({statistics['mean_absolute_error_percent']:.3f}%p)")
        print(f"  중앙값 오차율: {statistics['median_absolute_error']:.6f} ({statistics['median_absolute_error_percent']:.3f}%p)")
        print(f"  표준편차: {statistics['std_error']:.6f} ({statistics['std_error_percent']:.3f}%p)")
        print(f"  최소 오차: {statistics['min_error']:.6f} ({statistics['min_error']*100:.3f}%p)")
        print(f"  최대 오차: {statistics['max_error']:.6f} ({statistics['max_error']*100:.3f}%p)")
        print(f"  Q25: {statistics['q25_error']:.6f}")
        print(f"  Q75: {statistics['q75_error']:.6f}")
        print(f"\n[상대 오차]")
        print(f"  평균 상대 오차율: {statistics['mean_relative_error_percent']:.2f}%")
        print(f"{'='*80}\n")
        
        # 오차가 큰 상위 5개 샘플
        sorted_results = sorted(results, key=lambda x: x['error'], reverse=True)
        print(f"오차가 큰 상위 5개 샘플:")
        for i, r in enumerate(sorted_results[:5], 1):
            print(f"  {i}. 샘플 #{r['index']}: 실제={r['actual_rate']:.4f}, 예측={r['predicted_rate']:.4f}, "
                  f"오차={r['error_percent']:.2f}%p ({r['relative_error']:.1f}%)")
        
        # 오차가 작은 상위 5개 샘플
        print(f"\n오차가 작은 상위 5개 샘플:")
        sorted_best = sorted(results, key=lambda x: x['error'])
        for i, r in enumerate(sorted_best[:5], 1):
            print(f"  {i}. 샘플 #{r['index']}: 실제={r['actual_rate']:.4f}, 예측={r['predicted_rate']:.4f}, "
                  f"오차={r['error_percent']:.2f}%p ({r['relative_error']:.1f}%)")
        
        return {
            'statistics': statistics,
            'detailed_results': results,
            'test_data_path': test_data_path,
            'bin_width': bin_width
        }


def main():
    """사용 예시"""
    print("=" * 80)
    print("TFT 4-Feature 모델 - 가장 확률이 높은 구간 예측")
    print("=" * 80)
    
    predictor = ProbabilityPredictor(model_path='./results_tft_4feat/best_model.pt')
    
    # 예시 입력값
    input_dict = {
        '예가범위': 0.02,
        '낙찰하한율': 0.9,
        '추정가격': 53643620,
        '기초금액': 48279258
    }
    
    print(f"\n입력 피처:")
    for key, value in input_dict.items():
        print(f"  {key}: {value}")
    
    # 확률이 높은 상위 5개 구간
    result = predictor.get_highest_probability_ranges(input_dict, bin_width=0.001, top_k=5)
    
    print("\n" + "=" * 80)
    print(f"모델 예측 범위: {result['prediction_range']['min']*100:.2f}% ~ {result['prediction_range']['max']*100:.2f}%")
    print(f"중앙값: {result['statistics']['median']*100:.2f}%")
    print(f"평균: {result['statistics']['mean']*100:.2f}%")
    print("=" * 80)
    
    print("\n 사정률에 대한 구간별 확률")
    print(f"\n✨ 확률이 높은 상위 5개 구간:")
    for i, r in enumerate(result['top_ranges'], 1):
        print(f"  {i}위. {r['range']} = 사정율 {r['lower']*100:.1f}%~{r['upper']*100:.1f}% (확률: {r['probability_percent']:.2f}%)")
    
    # 전체 테스트 데이터에 대한 평가 (10,000개 샘플)
    print("\n\n" + "=" * 80)
    print("전체 테스트 데이터 평가 (최대 10,000개 샘플)")
    print("=" * 80)
    evaluation_result = predictor.evaluate_highest_probability_average(
        test_data_path='../dataset/dataset_feature_selected.csv',
        bin_width=0.001,
        max_samples=50000
    )


if __name__ == "__main__":
    print(f"Using device: {device}")
    main()

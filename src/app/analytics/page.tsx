"use client";

import styles from './page.module.css';

export default function Analytics() {
    const handleOpenApp = () => {
        window.open('https://kasancdss-isdoseprediction.streamlit.app/', '_blank', 'noopener,noreferrer');
    };

    return (
        <div className={styles.container}>
            <div className={styles.content}>
                {/* Header Section */}
                <div className={styles.header}>
                    <h1 className={styles.title}>IS Dose Prediction</h1>
                    <p className={styles.subtitle}>Immunosuppressant Dose Prediction System</p>
                </div>

                {/* Description Section */}
                <div className={styles.description}>
                    <h2 className={styles.sectionTitle}>About This Tool</h2>
                    <p className={styles.text}>
                        우리의 AI 기반 면역억제제 용량 예측 시스템은 신장 이식 환자를 위한
                        Tacrolimus 용량을 정확하게 예측합니다.
                    </p>
                    <p className={styles.text}>
                        이 도구는 환자의 임상 데이터와 최신 머신러닝 알고리즘을 활용하여
                        개인 맞춤형 용량 추천을 제공합니다.
                    </p>
                </div>

                {/* Features Section */}
                <div className={styles.features}>
                    <h2 className={styles.sectionTitle}>Key Features</h2>
                    <div className={styles.featureGrid}>
                        <div className={styles.featureCard}>
                            <div className={styles.featureIcon}>🎯</div>
                            <h3 className={styles.featureTitle}>정확한 예측</h3>
                            <p className={styles.featureText}>
                                Deep learning 모델을 통한 높은 정확도의 용량 예측
                            </p>
                        </div>
                        <div className={styles.featureCard}>
                            <div className={styles.featureIcon}>👤</div>
                            <h3 className={styles.featureTitle}>개인화</h3>
                            <p className={styles.featureText}>
                                환자별 특성을 반영한 맞춤형 용량 추천
                            </p>
                        </div>
                        <div className={styles.featureCard}>
                            <div className={styles.featureIcon}>📊</div>
                            <h3 className={styles.featureTitle}>실시간 분석</h3>
                            <p className={styles.featureText}>
                                TDM 결과를 바탕으로 즉시 용량 조정 제안
                            </p>
                        </div>
                        <div className={styles.featureCard}>
                            <div className={styles.featureIcon}>🔒</div>
                            <h3 className={styles.featureTitle}>안전성</h3>
                            <p className={styles.featureText}>
                                임상적으로 검증된 안전한 용량 범위 제공
                            </p>
                        </div>
                    </div>
                </div>

                {/* How to Use Section */}
                <div className={styles.howToUse}>
                    <h2 className={styles.sectionTitle}>How to Use</h2>
                    <ol className={styles.stepList}>
                        <li className={styles.step}>환자 정보 및 임상 데이터 입력</li>
                        <li className={styles.step}>이전 투약 기록 및 TDM 결과 입력</li>
                        <li className={styles.step}>AI 모델을 통한 용량 예측 실행</li>
                        <li className={styles.step}>추천된 용량 확인 및 임상적 판단 적용</li>
                    </ol>
                </div>

                {/* CTA Button */}
                <div className={styles.ctaSection}>
                    <button
                        onClick={handleOpenApp}
                        className={styles.ctaButton}
                    >
                        🚀 Streamlit 앱 열기
                    </button>
                    <p className={styles.ctaNote}>
                        * 새 창에서 Streamlit 애플리케이션이 실행됩니다
                    </p>
                </div>
            </div>
        </div>
    );
}

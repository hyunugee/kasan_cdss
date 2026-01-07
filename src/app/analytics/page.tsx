import styles from './page.module.css';

export default function Analytics() {
    return (
        <div className="container">
            <h1 className={styles.pageTitle}>Immunosuppressant Dose Prediction</h1>
            <p className={styles.intro}>
                Tacrolimus Concentration Prediction
            </p>

            <div className={styles.dashboard}>
                {/* Input Section */}
                <div className={`card ${styles.inputCard}`}>
                    <h2 className={styles.cardTitle}>Patient Parameters</h2>
                    <form className={styles.form}>
                        <div className={styles.formGroup}>
                            <label>Age</label>
                            <input type="number" placeholder="45" />
                        </div>
                        <div className={styles.formGroup}>
                            <label>Weight (kg)</label>
                            <input type="number" placeholder="70" />
                        </div>
                        <div className={styles.formGroup}>
                            <label>CYP3A5 Genotype</label>
                            <select>
                                <option>*1/*1</option>
                                <option>*1/*3</option>
                                <option>*3/*3</option>
                            </select>
                        </div>
                        <div className={styles.formGroup}>
                            <label>Current Dose (mg)</label>
                            <input type="number" step="0.5" placeholder="3.0" />
                        </div>
                        <button type="button" className="btn btn-primary" style={{ marginTop: '1rem', width: '100%' }}>
                            Run Prediction Model
                        </button>
                    </form>
                </div>

                {/* Results Section */}
                <div className={`card ${styles.resultCard}`}>
                    <h2 className={styles.cardTitle}>Prediction Results</h2>

                    <div className={styles.predictionBox}>
                        <div className={styles.label}>Predicted Trough Level</div>
                        <div className={styles.value}>
                            7.2 <span className={styles.unit}>ng/mL</span>
                        </div>
                        <div className={styles.status}>
                            <span className={styles.dotSuccess}></span> Optimal Range (6-8 ng/mL)
                        </div>
                    </div>

                    <div className={styles.divider}></div>

                    <div className={styles.prognosis}>
                        <div className={styles.label}>5-Year Graft Survival Probability</div>
                        <div className={styles.survivalBar}>
                            <div className={styles.survivalFill} style={{ width: '92%' }}>92%</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

"use client";

import { useState } from 'react';
import styles from './page.module.css';

// Types
interface PatientFeatures {
    AGE?: number;
    Sex?: string;
    Bwt?: number;
    Ht?: number;
    BMI?: number;
    Cause?: string;
    HD_type?: string;
    HD_duration?: number;
    DM?: string;
    HTN?: string;
}

interface DayData {
    prev_pm: number | null;
    am: number | null;
    tdm: number | null;
    predicted_pm: number | null; // Result of PM model (today PM dose)
    predicted_am: number | null; // Result of AM model (next day AM dose)
}

export default function Analytics() {
    // --- State ---
    // Patient Info
    const [patientId, setPatientId] = useState('');
    const [patientName, setPatientName] = useState('');
    const [confirmedPatient, setConfirmedPatient] = useState(false);

    // Static Features
    const [staticFeatures, setStaticFeatures] = useState<PatientFeatures>({});

    // Day Data (1-8)
    const [daysData, setDaysData] = useState<Record<number, DayData>>({
        1: { prev_pm: 0, am: 0, tdm: 0, predicted_pm: null, predicted_am: null },
        2: { prev_pm: 0, am: 0, tdm: 0, predicted_pm: null, predicted_am: null },
        3: { prev_pm: 0, am: 0, tdm: 0, predicted_pm: null, predicted_am: null },
        4: { prev_pm: 0, am: 0, tdm: 0, predicted_pm: null, predicted_am: null },
        5: { prev_pm: 0, am: 0, tdm: 0, predicted_pm: null, predicted_am: null },
        6: { prev_pm: 0, am: 0, tdm: 0, predicted_pm: null, predicted_am: null },
        7: { prev_pm: 0, am: 0, tdm: 0, predicted_pm: null, predicted_am: null },
        8: { prev_pm: 0, am: 0, tdm: 0, predicted_pm: null, predicted_am: null },
    });

    // UI State
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // --- Helpers ---
    const handleFeatureChange = (key: keyof PatientFeatures, value: string | number) => {
        setStaticFeatures(prev => ({ ...prev, [key]: value }));
    };

    const calculateFeatureCount = () => {
        // Count how many keys have non-empty values
        return Object.entries(staticFeatures).filter(([_, v]) => v !== '' && v !== undefined && v !== null).length;
    };

    const handleConfirmPatient = () => {
        if (!patientId || !patientName) {
            setError("Please enter patient ID and name.");
            return;
        }
        setError(null);
        setLoading(true);
        // Simulate loading
        setTimeout(() => {
            setLoading(false);
            setConfirmedPatient(true);
        }, 500);
    };

    const handleReset = () => {
        if (confirm("Are you sure you want to reset all data?")) {
            setConfirmedPatient(false);
            setPatientId('');
            setPatientName('');
            setStaticFeatures({});
            setDaysData({
                1: { prev_pm: 0, am: 0, tdm: 0, predicted_pm: null, predicted_am: null },
                2: { prev_pm: 0, am: 0, tdm: 0, predicted_pm: null, predicted_am: null },
                3: { prev_pm: 0, am: 0, tdm: 0, predicted_pm: null, predicted_am: null },
                4: { prev_pm: 0, am: 0, tdm: 0, predicted_pm: null, predicted_am: null },
                5: { prev_pm: 0, am: 0, tdm: 0, predicted_pm: null, predicted_am: null },
                6: { prev_pm: 0, am: 0, tdm: 0, predicted_pm: null, predicted_am: null },
                7: { prev_pm: 0, am: 0, tdm: 0, predicted_pm: null, predicted_am: null },
                8: { prev_pm: 0, am: 0, tdm: 0, predicted_pm: null, predicted_am: null },
            });
        }
    };

    const handleDayDataChange = (day: number, field: keyof DayData, value: number) => {
        setDaysData(prev => ({
            ...prev,
            [day]: {
                ...prev[day],
                [field]: value
            }
        }));
    };

    const runPrediction = async (day: number) => {
        const data = daysData[day];
        if (!data.prev_pm || !data.am || !data.tdm) {
            setError(`Day ${day}: Please enter previous PM dose, today AM dose, and FK TDM.`);
            return;
        }
        setError(null);
        setLoading(true);

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    target_day: day,
                    patient_data: daysData,
                    static_features: staticFeatures
                })
            });

            const result = await response.json();

            if (!response.ok || result.error) {
                throw new Error(result.error || result.details || 'Prediction failed');
            }

            const { predicted_pm, predicted_am } = result;

            setDaysData(prev => {
                const newData = { ...prev };

                // Update current day prediction results
                newData[day] = {
                    ...newData[day],
                    predicted_pm: predicted_pm
                };

                // Update next day inputs automatically
                if (day < 8) {
                    newData[day + 1] = {
                        ...newData[day + 1],
                        prev_pm: predicted_pm,
                        am: predicted_am,
                        predicted_am: predicted_am // Store for display
                    };
                }
                return newData;
            });

        } catch (err: any) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };


    // --- Render ---

    return (
        <div className={styles.container}>
            {/* Header */}
            <div className={styles.header}>
                <img src="/mark.png" alt="Logo" className={styles.logo} />
                <span className={styles.headerTitle}>ASAN MEDICAL</span>
            </div>

            {/* Patient Info Section */}
            <div className={styles.sectionHeader}>Patient information</div>

            <div className={styles.row}>
                <div className={styles.col}>
                    <div className={styles.inputGroup}>
                        <label className={styles.label}>Patient ID</label>
                        <input
                            className={styles.input}
                            value={patientId}
                            onChange={(e) => setPatientId(e.target.value)}
                            disabled={confirmedPatient}
                        />
                    </div>
                </div>
                <div className={styles.col}>
                    <div className={styles.inputGroup}>
                        <label className={styles.label}>Patient Name</label>
                        <input
                            className={styles.input}
                            value={patientName}
                            onChange={(e) => setPatientName(e.target.value)}
                            disabled={confirmedPatient}
                        />
                    </div>
                </div>
            </div>

            {/* Static Features Section */}
            <div className={styles.subHeader}>Patient Features (Optional)</div>
            <div className={styles.caption}>
                If you provide all patient features, a patient-specific time-series model is used; otherwise a general time-series model is used.
            </div>

            {/* Row 1 */}
            <div className={styles.row}>
                <div className={styles.colSmall}>
                    <div className={styles.inputGroup}>
                        <label className={styles.label}>Age</label>
                        <input type="number" className={styles.input} value={staticFeatures.AGE || ''} onChange={(e) => handleFeatureChange('AGE', parseFloat(e.target.value))} disabled={confirmedPatient} />
                    </div>
                </div>
                <div className={styles.colSmall}>
                    <div className={styles.inputGroup}>
                        <label className={styles.label}>Sex</label>
                        <select className={styles.select} value={staticFeatures.Sex || ''} onChange={(e) => handleFeatureChange('Sex', e.target.value)} disabled={confirmedPatient}>
                            <option value="">Select...</option>
                            <option value="Male">Male</option>
                            <option value="Female">Female</option>
                        </select>
                    </div>
                </div>
                <div className={styles.colSmall}>
                    <div className={styles.inputGroup}>
                        <label className={styles.label}>Body Weight (kg)</label>
                        <input type="number" className={styles.input} value={staticFeatures.Bwt || ''} onChange={(e) => handleFeatureChange('Bwt', parseFloat(e.target.value))} disabled={confirmedPatient} />
                    </div>
                </div>
                <div className={styles.colSmall}>
                    <div className={styles.inputGroup}>
                        <label className={styles.label}>Height (cm)</label>
                        <input type="number" className={styles.input} value={staticFeatures.Ht || ''} onChange={(e) => handleFeatureChange('Ht', parseFloat(e.target.value))} disabled={confirmedPatient} />
                    </div>
                </div>
                <div className={styles.colSmall}>
                    <div className={styles.inputGroup}>
                        <label className={styles.label}>BMI</label>
                        <input type="number" className={styles.input} value={staticFeatures.BMI || ''} onChange={(e) => handleFeatureChange('BMI', parseFloat(e.target.value))} disabled={confirmedPatient} />
                    </div>
                </div>
            </div>

            {/* Row 2 */}
            <div className={styles.row}>
                <div className={styles.colSmall}>
                    <div className={styles.inputGroup}>
                        <label className={styles.label}>Cause</label>
                        <select className={styles.select} value={staticFeatures.Cause || ''} onChange={(e) => handleFeatureChange('Cause', e.target.value)} disabled={confirmedPatient}>
                            <option value="">Select...</option>
                            <option value="HTN">HTN</option>
                            <option value="DM">DM</option>
                            <option value="GN">GN</option>
                            <option value="IgA">IgA</option>
                            <option value="FSGS">FSGS</option>
                            <option value="PCKD">PCKD</option>
                            <option value="Unknown">Unknown</option>
                            <option value="etc.">etc.</option>
                        </select>
                    </div>
                </div>
                <div className={styles.colSmall}>
                    <div className={styles.inputGroup}>
                        <label className={styles.label}>Hemodialysis Type</label>
                        <select className={styles.select} value={staticFeatures.HD_type || ''} onChange={(e) => handleFeatureChange('HD_type', e.target.value)} disabled={confirmedPatient}>
                            <option value="">Select...</option>
                            <option value="Preemptive">Preemptive</option>
                            <option value="HD">HD</option>
                            <option value="CAPD">CAPD</option>
                            <option value="HD+PD">HD+PD</option>
                        </select>
                    </div>
                </div>
                <div className={styles.colSmall}>
                    <div className={styles.inputGroup}>
                        <label className={styles.label}>HD Duration (months)</label>
                        <input type="number" className={styles.input} value={staticFeatures.HD_duration || ''} onChange={(e) => handleFeatureChange('HD_duration', parseFloat(e.target.value))} disabled={confirmedPatient} />
                    </div>
                </div>
                <div className={styles.colSmall}>
                    <div className={styles.inputGroup}>
                        <label className={styles.label}>Diabetes Mellitus</label>
                        <select className={styles.select} value={staticFeatures.DM || ''} onChange={(e) => handleFeatureChange('DM', e.target.value)} disabled={confirmedPatient}>
                            <option value="">Select...</option>
                            <option value="No">No</option>
                            <option value="Yes">Yes</option>
                        </select>
                    </div>
                </div>
                <div className={styles.colSmall}>
                    <div className={styles.inputGroup}>
                        <label className={styles.label}>Hypertension</label>
                        <select className={styles.select} value={staticFeatures.HTN || ''} onChange={(e) => handleFeatureChange('HTN', e.target.value)} disabled={confirmedPatient}>
                            <option value="">Select...</option>
                            <option value="No">No</option>
                            <option value="Yes">Yes</option>
                        </select>
                    </div>
                </div>
            </div>

            {/* Feature Status Info */}
            {!confirmedPatient && (
                calculateFeatureCount() === 10 ? (
                    <div className={styles.successBox}>✅ All patient features are provided: using a patient-specific time-series model.</div>
                ) : calculateFeatureCount() > 0 ? (
                    <div className={styles.warningBox}>⚠️ Only {calculateFeatureCount()}/10 patient features are provided: using a general time-series model.</div>
                ) : (
                    <div className={styles.infoBox}>ℹ️ No patient features provided.</div>
                )
            )}

            {/* Confirm Button */}
            {!confirmedPatient ? (
                <button className={styles.buttonPrimary} onClick={handleConfirmPatient} disabled={loading}>
                    {loading ? 'Loading...' : 'Confirm'}
                </button>
            ) : (
                <>
                    <div className={styles.divider} />
                    <div className={styles.row} style={{ alignItems: 'center' }}>
                        <div className={styles.col} style={{ flex: 4 }}>
                            <div className={styles.infoBox} style={{ margin: 0 }}>
                                <strong>Current patient</strong>: {patientId} - {patientName} | <strong>Model</strong>: {calculateFeatureCount() === 10 ? "Patient-specific" : "General"}
                            </div>
                        </div>
                        <div className={styles.col} style={{ flex: 1 }}>
                            <button className={styles.buttonSecondary} onClick={handleReset}>🗑️ Reset data</button>
                        </div>
                    </div>

                    <div className={styles.subHeader}>FK dose prediction</div>

                    {error && <div style={{ color: 'red', marginBottom: '1rem' }}>{error}</div>}

                    {/* Day Expanders */}
                    {[1, 2, 3, 4, 5, 6, 7, 8].map(day => (
                        <DayExpander
                            key={day}
                            day={day}
                            data={daysData[day]}
                            nextDayAm={day < 8 ? daysData[day + 1]?.predicted_am : null}
                            onChange={handleDayDataChange}
                            onPredict={runPrediction}
                            loading={loading}
                        />
                    ))}

                    <div className={styles.divider} />
                    <div className={styles.subHeader}>📊 Summary table</div>

                    <div className={styles.tableWrapper}>
                        <table className={styles.table}>
                            <thead>
                                <tr>
                                    <th>Day</th>
                                    <th>전날 오후 FK용량 (Previous PM FK dose)</th>
                                    <th>당일 오전 FK용량 (Today AM FK dose)</th>
                                    <th>FK TDM (FK trough level)</th>
                                </tr>
                            </thead>
                            <tbody>
                                {[1, 2, 3, 4, 5, 6, 7, 8].map(day => (
                                    <tr key={day}>
                                        <td>Day {day}</td>
                                        <td>{daysData[day].prev_pm || ''}</td>
                                        <td>{daysData[day].am || ''}</td>
                                        <td>{daysData[day].tdm || ''}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>

                </>
            )}

        </div>
    );
}

// Subcomponent for Day Expander
function DayExpander({ day, data, nextDayAm, onChange, onPredict, loading }: {
    day: number,
    data: DayData,
    nextDayAm: number | null | undefined, // optional for Day 8
    onChange: (day: number, field: keyof DayData, value: number) => void,
    onPredict: (day: number) => void,
    loading: boolean
}) {
    const [expanded, setExpanded] = useState(day <= 3);

    return (
        <div className={styles.expander}>
            <div className={styles.expanderSummary} onClick={() => setExpanded(!expanded)}>
                <span>Day {day}</span>
                <span>{expanded ? '−' : '+'}</span>
            </div>
            {expanded && (
                <div className={styles.expanderContent}>
                    <div className={styles.row}>
                        {/* Input 1 */}
                        <div className={styles.col}>
                            <div className={styles.inputGroup}>
                                <div style={{ fontSize: '0.9rem', fontWeight: 700, marginBottom: '0.25rem', color: '#31333F' }}>전날 오후 FK용량</div>
                                <label className={styles.label}>Previous PM dose (mg)</label>
                                <input
                                    type="number"
                                    step="0.05"
                                    className={styles.input}
                                    value={data.prev_pm || ''}
                                    onChange={(e) => onChange(day, 'prev_pm', parseFloat(e.target.value))}
                                />
                            </div>
                        </div>

                        {/* Input 2 */}
                        <div className={styles.col}>
                            <div className={styles.inputGroup}>
                                <div style={{ fontSize: '0.9rem', fontWeight: 700, marginBottom: '0.25rem', color: '#31333F' }}>당일 오전 FK용량</div>
                                <label className={styles.label}>Today AM dose (mg)</label>
                                <input
                                    type="number"
                                    step="0.05"
                                    className={styles.input}
                                    value={data.am || ''}
                                    onChange={(e) => onChange(day, 'am', parseFloat(e.target.value))}
                                />
                            </div>
                        </div>

                        {/* Input 3 */}
                        <div className={styles.col}>
                            <div className={styles.inputGroup}>
                                <div style={{ fontSize: '0.9rem', fontWeight: 700, marginBottom: '0.25rem', color: '#31333F' }}>FK TDM</div>
                                <label className={styles.label}>FK TDM level</label>
                                <input
                                    type="number"
                                    step="0.1"
                                    className={styles.input}
                                    value={data.tdm || ''}
                                    onChange={(e) => onChange(day, 'tdm', parseFloat(e.target.value))}
                                />
                            </div>
                        </div>

                        {/* Prediction Results */}
                        <div className={styles.col}>
                            <div className={styles.inputGroup}>
                                <div style={{ fontSize: '0.9rem', fontWeight: 700, marginBottom: '0.25rem', color: '#31333F' }}>Prediction results</div>
                                <label className={styles.label}>&nbsp;</label>
                                {/* ^ Placeholder to align with labels */}

                                {data.predicted_pm !== null ? (
                                    <div className={styles.predictionResult}>
                                        Today PM: <strong>{data.predicted_pm.toFixed(2)}</strong> mg
                                    </div>
                                ) : (
                                    <div style={{ color: '#aaa', fontSize: '0.9rem', padding: '0.5rem 0' }}>_No prediction yet_</div>
                                )}

                                {nextDayAm !== null && nextDayAm !== undefined && (
                                    <div className={styles.predictionResult}>
                                        Next day AM: <strong>{nextDayAm.toFixed(2)}</strong> mg
                                    </div>
                                )}
                            </div>
                        </div>

                        {/* Run Button */}
                        <div className={styles.col} style={{ display: 'flex', alignItems: 'flex-end' }}>
                            <button className={styles.buttonPrimary} onClick={() => onPredict(day)} disabled={loading}>
                                {loading ? 'Predicting...' : '🔮 Predict'}
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

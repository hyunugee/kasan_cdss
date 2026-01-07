export default function LabInfo() {
    return (
        <div className="container">
            <div style={{ maxWidth: '800px', margin: '3rem auto' }}>
                <h1 style={{ fontSize: '2.5rem', fontWeight: '800', marginBottom: '1.5rem', textAlign: 'center' }}>
                    Lab Info & Brand Story
                </h1>

                <div className="card" style={{ marginBottom: '2rem' }}>
                    <h2 style={{ fontSize: '1.5rem', fontWeight: '700', marginBottom: '1rem', color: 'var(--color-orange-primary)' }}>
                        About KASAN
                    </h2>
                    <p style={{ lineHeight: '1.8', color: '#444', marginBottom: '1rem' }}>
                        The <strong>Kidney Allograft Survival Analysis Network (KASAN)</strong> is a pioneering research initiative dedicated to improving the long-term outcomes of kidney and pancreas transplant recipients.
                    </p>
                    <p style={{ lineHeight: '1.8', color: '#444' }}>
                        Our mission is to bridge the gap between complex clinical data and actionable medical insights using advanced Artificial Intelligence. By analyzing vast datasets of patient records, genetic markers, and drug responses, KASAN aims to predict graft survival and optimize immunosuppressive therapy with unprecedented accuracy.
                    </p>
                </div>

                <div className="card">
                    <h2 style={{ fontSize: '1.5rem', fontWeight: '700', marginBottom: '1rem', color: 'var(--color-green-primary)' }}>
                        Our Philosophy
                    </h2>
                    <p style={{ lineHeight: '1.8', color: '#444', marginBottom: '1.5rem' }}>
                        KASAN represents the duality of transplant care:
                    </p>
                    <ul style={{ listStyle: 'none', padding: 0 }}>
                        <li style={{ marginBottom: '1rem', display: 'flex', gap: '1rem' }}>
                            <span style={{
                                display: 'inline-block', width: '12px', height: '12px',
                                borderRadius: '50%', backgroundColor: 'var(--color-orange-primary)', marginTop: '8px'
                            }}></span>
                            <div>
                                <strong>Energy & Warning (Orange)</strong><br />
                                Vigilance in monitoring, rapid response to rejection risks, and the vibrant energy of life restored.
                            </div>
                        </li>
                        <li style={{ display: 'flex', gap: '1rem' }}>
                            <span style={{
                                display: 'inline-block', width: '12px', height: '12px',
                                borderRadius: '50%', backgroundColor: 'var(--color-green-primary)', marginTop: '8px'
                            }}></span>
                            <div>
                                <strong>Stability & Recovery (Green)</strong><br />
                                Long-term graft survival, balance in medication, and the peace of mind for patients returning to daily life.
                            </div>
                        </li>
                    </ul>
                </div>
            </div>
        </div>
    );
}

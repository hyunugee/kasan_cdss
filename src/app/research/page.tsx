export default function Research() {
    const papers = [
        {
            year: 2024,
            title: "Machine Learning-Based Prediction of Tacrolimus Stable Dose in Kidney Transplant Recipients",
            journal: "Journal of Clinical Medicine",
            authors: "Kim M.S., Park J.B., et al."
        },
        {
            year: 2023,
            title: "Long-term Graft Survival Analysis Using Deep Learning Models",
            journal: "Transplantation Proceedings",
            authors: "Lee Y.J., Park J.B., et al."
        },
        {
            year: 2023,
            title: "Impact of CYP3A5 Genotype on Tacrolimus Concentration",
            journal: "Pharmacogenomics Journal",
            authors: "Park H.W., Kim M.S., et al."
        }
    ];

    return (
        <div className="container">
            <div style={{ maxWidth: '800px', margin: '3rem auto' }}>
                <h1 style={{ fontSize: '2.5rem', fontWeight: '800', marginBottom: '2rem' }}>Research & Pubs</h1>

                <section style={{ marginBottom: '3rem' }}>
                    <h2 style={{ fontSize: '1.5rem', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <span style={{ width: '4px', height: '24px', backgroundColor: 'var(--color-orange-primary)' }}></span>
                        Active Projects
                    </h2>
                    <div className="card" style={{ marginBottom: '1rem' }}>
                        <h3 style={{ fontWeight: '700', marginBottom: '0.5rem' }}>KASAN-AI Project</h3>
                        <p style={{ color: '#666', lineHeight: '1.6' }}>
                            Developing a comprehensive AI model for real-time immunosuppressant monitoring and rejection prediction. Features include integration with AWS SageMaker for continuous model training.
                        </p>
                    </div>
                </section>

                <section>
                    <h2 style={{ fontSize: '1.5rem', marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <span style={{ width: '4px', height: '24px', backgroundColor: 'var(--color-green-primary)' }}></span>
                        Selected Publications
                    </h2>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                        {papers.map((paper, idx) => (
                            <div key={idx} className="card" style={{ padding: '1.25rem' }}>
                                <div style={{ fontSize: '0.9rem', color: '#888', marginBottom: '0.25rem' }}>{paper.year} | {paper.journal}</div>
                                <h3 style={{ fontSize: '1.1rem', fontWeight: '600', marginBottom: '0.5rem', lineHeight: '1.4' }}>
                                    {paper.title}
                                </h3>
                                <div style={{ fontSize: '0.9rem', color: '#555' }}>{paper.authors}</div>
                            </div>
                        ))}
                    </div>
                </section>
            </div>
        </div>
    );
}

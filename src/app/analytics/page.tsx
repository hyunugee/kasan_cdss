"use client";

export default function Analytics() {
    return (
        <div style={{
            width: '100%',
            height: 'calc(100vh - 80px)', // Adjust based on your navbar height
            padding: 0,
            margin: 0
        }}>
            <iframe
                src="https://kasancdss-isdoseprediction.streamlit.app/"
                style={{
                    width: '100%',
                    height: '100%',
                    border: 'none',
                    display: 'block'
                }}
                title="IS Dose Prediction Streamlit App"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
            />
        </div>
    );
}

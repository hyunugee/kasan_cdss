import styles from './page.module.css';

export default function People() {
    const team = [
        { name: "Prof. Hyunwook Kwon", role: "Principal Investigator", specialty: "Transplant Surgery" },
        { name: "Dr. ", role: "Lead AI Researcher", specialty: "Machine Learning" },
        { name: "", role: "Clinical Coordinator", specialty: "Patient Care" },
        { name: "", role: "Data Scientist", specialty: "Bioinformatics" },
    ];

    return (
        <div className="container">
            <h1 className={styles.title}>People</h1>
            <p className={styles.subtitle}>The team behind KASAN</p>

            <div className={styles.grid}>
                {team.map((person, idx) => (
                    <div key={idx} className={`card ${styles.profileCard}`}>
                        <div className={styles.avatarPlaceholder}>
                            {person.name.split(' ').map(n => n[0]).join('').slice(0, 2)}
                        </div>
                        <h3 className={styles.name}>{person.name}</h3>
                        <div className={styles.role}>{person.role}</div>
                        <div className={styles.specialty}>{person.specialty}</div>
                    </div>
                ))}
            </div>
        </div>
    );
}

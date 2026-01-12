import styles from './page.module.css';

export default function Contact() {
    return (
        <div className="container">
            <h1 className={styles.title}>Contact Us</h1>
            <p className={styles.subtitle}>
                Interested in research collaboration or have questions? Get in touch.
            </p>

            <div className={styles.wrapper}>
                <div className={`card ${styles.formCard}`}>
                    <form className={styles.form}>
                        <div className={styles.formGroup}>
                            <label>Name</label>
                            <input type="text" placeholder="Your name" />
                        </div>
                        <div className={styles.formGroup}>
                            <label>Email</label>
                            <input type="email" placeholder="email@example.com" />
                        </div>
                        <div className={styles.formGroup}>
                            <label>Subject</label>
                            <select>
                                <option>Research Collaboration</option>
                                <option>Technical Support</option>
                                <option>General Inquiry</option>
                            </select>
                        </div>
                        <div className={styles.formGroup}>
                            <label>Message</label>
                            <textarea rows={5} placeholder="How can we help?"></textarea>
                        </div>
                        <button type="button" className="btn btn-primary">Send Message</button>
                    </form>
                </div>

                <div className={styles.info}>
                    <div className="card">
                        <h3 className={styles.infoTitle}>Lab Location</h3>
                        <p className={styles.infoText}>
                            Asan Medical Center<br />
                            88 Olympicro 43gil, Songpa-gu<br />
                            Seoul, South Korea
                        </p>
                    </div>

                    <div className="card" style={{ marginTop: '1.5rem' }}>
                        <h3 className={styles.infoTitle}>Email</h3>
                        <p className={styles.infoText}>
                            hwkwon@amc.seoul.kr<br />
                            ultrapolymer@hotmail.com
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
}

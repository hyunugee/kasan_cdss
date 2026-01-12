import styles from './page.module.css';

export default function Assistant() {
    return (
        <div className="container">
            <div className={styles.header}>
                <h1 className={styles.title}>AI Assistant</h1>
                <p className={styles.subtitle}>Ask questions about post-transplant care</p>
            </div>

            <div className={`card ${styles.chatContainer}`}>
                <div className={styles.messageList}>

                    <div className={`${styles.message} ${styles.bot}`}>
                        <div className={styles.avatar}>AI</div>
                        <div className={styles.bubble}>
                            Hello! I am the KASAN AI Assistant. How can I help you with your medication or symptoms today?
                        </div>
                    </div>

                    <div className={`${styles.message} ${styles.user}`}>
                        <div className={styles.bubble}>
                            My Tacrolimus level was 8.5 yesterday. Is that safe?
                        </div>
                        <div className={styles.avatar}>Me</div>
                    </div>

                    <div className={`${styles.message} ${styles.bot}`}>
                        <div className={styles.avatar}>AI</div>
                        <div className={styles.bubble}>
                            A level of 8.5 ng/mL is generally considered within the therapeutic range (usually 5-10 ng/mL, depending on your post-op duration). However, please consult your doctor for personalized advice. Would you like to check the trend?
                        </div>
                    </div>

                </div>

                <div className={styles.inputArea}>
                    <input
                        type="text"
                        className={styles.input}
                        placeholder="Type your question..."
                    />
                    <button className="btn btn-primary">Send</button>
                </div>
            </div>
        </div>
    );
}

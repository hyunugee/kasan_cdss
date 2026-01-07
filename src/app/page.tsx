import styles from './page.module.css';

export default function Home() {
  return (
    <div className={styles.container}>
      <div className={styles.heroContent}>
        <h1 className={styles.mainTitle}>
          <div className={styles.wordLine}>
            Dr.<span className={styles.charK}>K</span>'s
          </div>
          <div className={styles.wordLine}>
            <span className={styles.charA1}>A</span>san
          </div>
          <div className={styles.wordLine}>
            <span className={styles.charS}>S</span>mart
          </div>
          <div className={styles.wordLine}>
            <span className={styles.charA2}>A</span>nalytics
          </div>
          <div className={styles.wordLine}>
            <span className={styles.charN}>N</span>etwork
          </div>
        </h1>

        <div className={styles.subTitle}>AI Lab</div>
      </div>
    </div>
  );
}

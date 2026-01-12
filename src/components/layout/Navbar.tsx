"use client";

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { NAV_ITEMS } from '@/lib/constants';
import styles from './Navbar.module.css';

export default function Navbar() {
    const pathname = usePathname();

    return (
        <nav className={styles.navbar}>
            <div className="container">
                <div className={styles.nabaWrapper}>
                    <Link href="/" className={styles.logoLink}>
                        <img src="/logo.png" alt="KASAN AI Lab" className={styles.logoImage} />
                    </Link>

                    <div className={styles.menu}>
                        {NAV_ITEMS.map((item) => {
                            const isExternal = item.href.startsWith('http');
                            const isActive = pathname === item.href;

                            if (isExternal) {
                                return (
                                    <a
                                        key={item.href}
                                        href={item.href}
                                        className={styles.navItem}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                    >
                                        {item.label}
                                    </a>
                                );
                            }

                            return (
                                <Link
                                    key={item.href}
                                    href={item.href}
                                    className={`${styles.navItem} ${isActive ? styles.active : ''}`}
                                >
                                    {item.label}
                                </Link>
                            );
                        })}
                    </div>
                </div>
            </div>
        </nav>
    );
}

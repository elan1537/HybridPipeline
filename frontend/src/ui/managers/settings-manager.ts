/**
 * SettingsManager - Manages application settings via cookies
 * Extracted from main.ts (lines 217-233)
 */

export class SettingsManager {
  /**
   * Set a cookie
   * @param name Cookie name
   * @param value Cookie value
   * @param days Expiration in days (default: 30)
   */
  static setCookie(name: string, value: string, days: number = 30): void {
    const expires = new Date();
    expires.setTime(expires.getTime() + (days * 24 * 60 * 60 * 1000));
    document.cookie = `${name}=${value};expires=${expires.toUTCString()};path=/`;
  }

  /**
   * Get a cookie value
   * @param name Cookie name
   * @returns Cookie value or null if not found
   */
  static getCookie(name: string): string | null {
    const nameEQ = name + "=";
    const ca = document.cookie.split(';');
    for (let i = 0; i < ca.length; i++) {
      let c = ca[i];
      while (c.charAt(0) === ' ') c = c.substring(1, c.length);
      if (c.indexOf(nameEQ) === 0) return c.substring(nameEQ.length, c.length);
    }
    return null;
  }

  /**
   * Delete a cookie
   * @param name Cookie name
   */
  static deleteCookie(name: string): void {
    document.cookie = `${name}=;expires=Thu, 01 Jan 1970 00:00:00 UTC;path=/`;
  }

  /**
   * Check if a cookie exists
   * @param name Cookie name
   */
  static hasCookie(name: string): boolean {
    return this.getCookie(name) !== null;
  }
}

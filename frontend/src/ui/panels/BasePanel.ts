/**
 * BasePanel - Abstract base class for UI panels
 * Provides common DOM manipulation utilities
 */

export abstract class BasePanel {
  /**
   * Get a DOM element by ID with type safety
   */
  protected getElement<T extends HTMLElement>(id: string): T | null {
    return document.getElementById(id) as T | null;
  }

  /**
   * Update text content of an element (null-safe)
   */
  protected updateText(element: HTMLElement | null, text: string): void {
    if (element) {
      element.textContent = text;
    }
  }

  /**
   * Update innerHTML of an element (null-safe)
   */
  protected updateHtml(element: HTMLElement | null, html: string): void {
    if (element) {
      element.innerHTML = html;
    }
  }

  /**
   * Set display visibility of an element (null-safe)
   */
  protected setVisible(element: HTMLElement | null, visible: boolean): void {
    if (element) {
      element.style.display = visible ? 'block' : 'none';
    }
  }

  /**
   * Add event listener to an element (null-safe)
   */
  protected addListener<K extends keyof HTMLElementEventMap>(
    element: HTMLElement | null,
    event: K,
    handler: (ev: HTMLElementEventMap[K]) => void
  ): void {
    element?.addEventListener(event, handler);
  }

  /**
   * Set value of an input element (null-safe)
   */
  protected setValue(element: HTMLInputElement | null, value: string): void {
    if (element) {
      element.value = value;
    }
  }

  /**
   * Get value of an input element (null-safe)
   */
  protected getValue(element: HTMLInputElement | null, defaultValue = ''): string {
    return element?.value ?? defaultValue;
  }

  /**
   * Check if a checkbox is checked (null-safe)
   */
  protected isChecked(element: HTMLInputElement | null): boolean {
    return element?.checked ?? false;
  }

  /**
   * Set checkbox checked state (null-safe)
   */
  protected setChecked(element: HTMLInputElement | null, checked: boolean): void {
    if (element) {
      element.checked = checked;
    }
  }

  /**
   * Set disabled state of an element (null-safe)
   */
  protected setDisabled(element: HTMLInputElement | HTMLButtonElement | null, disabled: boolean): void {
    if (element) {
      element.disabled = disabled;
    }
  }

  /**
   * Cleanup resources - must be implemented by subclasses
   */
  abstract cleanup(): void;
}

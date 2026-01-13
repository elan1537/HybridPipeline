/**
 * EventBus - Simple pub/sub event system for decoupled communication
 * CP3: Core infrastructure for system communication
 */

import { EventHandler, AppEvents } from "../types";

export class EventBus {
  private events: Map<string, Set<EventHandler>>;
  private onceHandlers: Map<string, Set<EventHandler>>;

  constructor() {
    this.events = new Map();
    this.onceHandlers = new Map();
  }

  /**
   * Subscribe to an event
   */
  on<K extends keyof AppEvents>(event: K, handler: EventHandler<AppEvents[K]>): () => void {
    if (!this.events.has(event)) {
      this.events.set(event, new Set());
    }
    this.events.get(event)!.add(handler);

    // Return unsubscribe function
    return () => this.off(event, handler);
  }

  /**
   * Subscribe to an event once (auto-unsubscribe after first call)
   */
  once<K extends keyof AppEvents>(event: K, handler: EventHandler<AppEvents[K]>): void {
    if (!this.onceHandlers.has(event)) {
      this.onceHandlers.set(event, new Set());
    }
    this.onceHandlers.get(event)!.add(handler);
  }

  /**
   * Unsubscribe from an event
   */
  off<K extends keyof AppEvents>(event: K, handler: EventHandler<AppEvents[K]>): void {
    const handlers = this.events.get(event);
    if (handlers) {
      handlers.delete(handler);
      if (handlers.size === 0) {
        this.events.delete(event);
      }
    }

    const onceHandlers = this.onceHandlers.get(event);
    if (onceHandlers) {
      onceHandlers.delete(handler);
      if (onceHandlers.size === 0) {
        this.onceHandlers.delete(event);
      }
    }
  }

  /**
   * Emit an event synchronously
   */
  emit<K extends keyof AppEvents>(event: K, data?: AppEvents[K]): void {
    // Regular handlers
    const handlers = this.events.get(event);
    if (handlers) {
      handlers.forEach((handler) => {
        try {
          handler(data);
        } catch (error) {
          console.error(`Error in event handler for "${event}":`, error);
        }
      });
    }

    // Once handlers
    const onceHandlers = this.onceHandlers.get(event);
    if (onceHandlers) {
      onceHandlers.forEach((handler) => {
        try {
          handler(data);
        } catch (error) {
          console.error(`Error in once event handler for "${event}":`, error);
        }
      });
      this.onceHandlers.delete(event);
    }
  }

  /**
   * Emit an event asynchronously
   */
  async emitAsync<K extends keyof AppEvents>(event: K, data?: AppEvents[K]): Promise<void> {
    const handlers = this.events.get(event);
    const onceHandlers = this.onceHandlers.get(event);

    const allHandlers = [
      ...(handlers ? Array.from(handlers) : []),
      ...(onceHandlers ? Array.from(onceHandlers) : []),
    ];

    await Promise.all(
      allHandlers.map(async (handler) => {
        try {
          await handler(data);
        } catch (error) {
          console.error(`Error in async event handler for "${event}":`, error);
        }
      })
    );

    if (onceHandlers) {
      this.onceHandlers.delete(event);
    }
  }

  /**
   * Remove all event listeners
   */
  clear(): void {
    this.events.clear();
    this.onceHandlers.clear();
  }

  /**
   * Get number of listeners for an event
   */
  listenerCount(event: keyof AppEvents): number {
    const regular = this.events.get(event)?.size || 0;
    const once = this.onceHandlers.get(event)?.size || 0;
    return regular + once;
  }
}

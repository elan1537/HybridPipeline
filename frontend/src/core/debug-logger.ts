/**
 * Debug logging system with toggle functionality
 * Only logs messages when debug mode is enabled
 */

export class DebugLogger {
    private static debugEnabled: boolean = false;
    
    static setDebugEnabled(enabled: boolean): void {
        this.debugEnabled = enabled;
        if (enabled) {
            console.log('🐛 Debug logging enabled');
        } else {
            console.log('🔇 Debug logging disabled');
            // Clear the console when debug logging is disabled
            console.clear();
        }
    }
    
    static isDebugEnabled(): boolean {
        return this.debugEnabled;
    }
    
    static log(message: string, ...args: any[]): void {
        if (this.debugEnabled) {
            console.log(`[DEBUG] ${message}`, ...args);
        }
    }
    
    static warn(message: string, ...args: any[]): void {
        if (this.debugEnabled) {
            console.warn(`[DEBUG WARN] ${message}`, ...args);
        }
    }
    
    static error(message: string, ...args: any[]): void {
        if (this.debugEnabled) {
            console.error(`[DEBUG ERROR] ${message}`, ...args);
        }
    }

    // Component-specific loggers
    static logMain(message: string, ...args: any[]): void {
        this.log(`[Main] ${message}`, ...args);
    }
    
    static logWorker(message: string, ...args: any[]): void {
        this.log(`[Worker] ${message}`, ...args);
    }
    
    static logLatency(message: string, ...args: any[]): void {
        this.log(`[LatencyTracker] ${message}`, ...args);
    }
    
    static logFPS(message: string, ...args: any[]): void {
        this.log(`[FPS] ${message}`, ...args);
    }
}

// Export a shorter alias for convenience
export const debug = DebugLogger;
/**
 * ControlPanel - Connection and rendering mode controls
 * Extracted from main.ts (lines 93-96, 146-150, 200-202, 281-331, 877-977)
 */

import { debug } from '../../debug-logger';
import { RenderMode } from '../../types';
import { BasePanel } from './BasePanel';

// Callbacks for handling UI events
export interface ControlPanelCallbacks {
  onConnect?: () => void;
  onDisconnect?: () => void;
  onEncoderChange?: (isJpeg: boolean) => void;
  onDepthDebugToggle?: (enabled: boolean) => void;
  onRenderModeChange?: (mode: RenderMode) => void;
}

// Render mode radio button configuration
interface RenderModeConfig {
  id: string;
  mode: RenderMode;
}

export class ControlPanel extends BasePanel {
  // DOM elements - Connection controls
  private wsConnectButton: HTMLInputElement | null = null;
  private wsDisconnectButton: HTMLInputElement | null = null;
  private wsStateConsoleText: HTMLDivElement | null = null;

  // DOM elements - Encoder control
  private jpegFallbackCheckbox: HTMLInputElement | null = null;

  // DOM elements - Render mode controls (stored as map for easy access)
  private renderModeRadios: Map<RenderMode, HTMLInputElement | null> = new Map();

  // DOM elements - Debug controls
  private depthDebugCheckbox: HTMLInputElement | null = null;

  // State
  private currentRenderMode: RenderMode = RenderMode.FUSION;
  private callbacks: ControlPanelCallbacks = {};

  // Render mode configuration
  private static readonly RENDER_MODES: RenderModeConfig[] = [
    { id: 'fusion-mode', mode: RenderMode.FUSION },
    { id: 'gaussian-only-mode', mode: RenderMode.GAUSSIAN_ONLY },
    { id: 'local-only-mode', mode: RenderMode.LOCAL_ONLY },
    { id: 'depth-fusion-mode', mode: RenderMode.DEPTH_FUSION },
    { id: 'feed-forward-mode', mode: RenderMode.FEED_FORWARD },
  ];

  constructor(callbacks: ControlPanelCallbacks = {}) {
    super();
    this.callbacks = callbacks;
    this.initializeElements();
    this.setupListeners();
  }

  private initializeElements(): void {
    // Connection controls
    this.wsConnectButton = this.getElement('ws-connect-button');
    this.wsDisconnectButton = this.getElement('ws-disconnect-button');
    this.wsStateConsoleText = this.getElement('ws-state-console-text');

    // Encoder control
    this.jpegFallbackCheckbox = this.getElement('jpeg-fallback-checkbox');

    // Render mode controls - initialize using configuration
    ControlPanel.RENDER_MODES.forEach(config => {
      this.renderModeRadios.set(config.mode, this.getElement(config.id));
    });

    // Debug controls
    this.depthDebugCheckbox = this.getElement('depth-debug-checkbox');

    if (!this.wsConnectButton || !this.wsDisconnectButton) {
      debug.warn('[ControlPanel] Connection buttons not found');
    }
  }

  private setupListeners(): void {
    // Connection buttons
    this.addListener(this.wsConnectButton, 'click', () => this.handleConnect());
    this.addListener(this.wsDisconnectButton, 'click', () => this.handleDisconnect());

    // Encoder toggle
    this.addListener(this.jpegFallbackCheckbox, 'click', () => this.handleEncoderChange());

    // Depth debug toggle
    this.addListener(this.depthDebugCheckbox, 'click', () => this.handleDepthDebugToggle());

    // Render mode radios - setup using configuration (loop-based)
    ControlPanel.RENDER_MODES.forEach(config => {
      const radio = this.renderModeRadios.get(config.mode);
      this.addListener(radio, 'change', () => {
        if (this.isChecked(radio)) {
          this.handleRenderModeChange(config.mode);
        }
      });
    });
  }

  // Event handlers (delegate to callbacks)
  private handleConnect(): void {
    debug.logMain('[ControlPanel] Connect button clicked');
    this.callbacks.onConnect?.();
  }

  private handleDisconnect(): void {
    debug.logMain('[ControlPanel] Disconnect button clicked');
    this.callbacks.onDisconnect?.();
  }

  private handleEncoderChange(): void {
    const isJpegMode = this.isChecked(this.jpegFallbackCheckbox);
    debug.logMain(`[ControlPanel] Encoder changed to ${isJpegMode ? 'JPEG' : 'H264'} mode`);
    this.callbacks.onEncoderChange?.(isJpegMode);
  }

  private handleDepthDebugToggle(): void {
    const isEnabled = this.isChecked(this.depthDebugCheckbox);
    debug.logMain(`[ControlPanel] Depth Debug: ${isEnabled ? 'Enabled' : 'Disabled'}`);
    this.callbacks.onDepthDebugToggle?.(isEnabled);
  }

  private handleRenderModeChange(mode: RenderMode): void {
    this.currentRenderMode = mode;
    debug.logMain(`[ControlPanel] Switched to ${mode} mode`);
    this.callbacks.onRenderModeChange?.(mode);
  }

  // Public methods
  setConnectionState(state: 'connected' | 'disconnected' | 'error' | 'closed'): void {
    const stateMap = {
      connected: 'WS State: Connected',
      disconnected: 'WS State: Disconnected',
      error: 'WS State: Error',
      closed: 'WS State: Closed'
    };

    this.updateText(this.wsStateConsoleText, stateMap[state]);
  }

  getRenderMode(): RenderMode {
    return this.currentRenderMode;
  }

  setRenderMode(mode: RenderMode): void {
    this.currentRenderMode = mode;

    // Update radio buttons using configuration
    ControlPanel.RENDER_MODES.forEach(config => {
      const radio = this.renderModeRadios.get(config.mode);
      this.setChecked(radio, config.mode === mode);
    });
  }

  isJpegMode(): boolean {
    return this.isChecked(this.jpegFallbackCheckbox);
  }

  setJpegMode(enabled: boolean): void {
    this.setChecked(this.jpegFallbackCheckbox, enabled);
  }

  isDepthDebugEnabled(): boolean {
    return this.isChecked(this.depthDebugCheckbox);
  }

  setCallbacks(callbacks: ControlPanelCallbacks): void {
    this.callbacks = { ...this.callbacks, ...callbacks };
  }

  cleanup(): void {
    // Event listeners are automatically removed when elements are removed from DOM
  }
}

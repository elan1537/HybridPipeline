/**
 * ControlPanel - Connection and rendering mode controls
 * Extracted from main.ts (lines 93-96, 146-150, 200-202, 281-331, 877-977)
 */

import { debug } from '../../debug-logger';

// Render mode enum (matches main.ts:153-159)
export enum RenderMode {
  FUSION = 'fusion',
  GAUSSIAN_ONLY = 'gaussian',
  LOCAL_ONLY = 'local',
  DEPTH_FUSION = 'depth-fusion',
  FEED_FORWARD = 'feed-forward'
}

// Callbacks for handling UI events
export interface ControlPanelCallbacks {
  onConnect?: () => void;
  onDisconnect?: () => void;
  onEncoderChange?: (isJpeg: boolean) => void;
  onDepthDebugToggle?: (enabled: boolean) => void;
  onRenderModeChange?: (mode: RenderMode) => void;
}

export class ControlPanel {
  // DOM elements - Connection controls
  private wsConnectButton: HTMLInputElement | null = null;
  private wsDisconnectButton: HTMLInputElement | null = null;
  private wsStateConsoleText: HTMLDivElement | null = null;

  // DOM elements - Encoder control
  private jpegFallbackCheckbox: HTMLInputElement | null = null;

  // DOM elements - Render mode controls
  private fusionModeRadio: HTMLInputElement | null = null;
  private gaussianOnlyModeRadio: HTMLInputElement | null = null;
  private localOnlyModeRadio: HTMLInputElement | null = null;
  private depthFusionModeRadio: HTMLInputElement | null = null;
  private feedForwardModeRadio: HTMLInputElement | null = null;

  // DOM elements - Debug controls
  private depthDebugCheckbox: HTMLInputElement | null = null;

  // State
  private currentRenderMode: RenderMode = RenderMode.FUSION;
  private callbacks: ControlPanelCallbacks = {};

  constructor(callbacks: ControlPanelCallbacks = {}) {
    this.callbacks = callbacks;
    this.initializeElements();
    this.setupListeners();
  }

  private initializeElements(): void {
    // Connection controls
    this.wsConnectButton = document.getElementById('ws-connect-button') as HTMLInputElement;
    this.wsDisconnectButton = document.getElementById('ws-disconnect-button') as HTMLInputElement;
    this.wsStateConsoleText = document.getElementById('ws-state-console-text') as HTMLDivElement;

    // Encoder control
    this.jpegFallbackCheckbox = document.getElementById('jpeg-fallback-checkbox') as HTMLInputElement;

    // Render mode controls
    this.fusionModeRadio = document.getElementById('fusion-mode') as HTMLInputElement;
    this.gaussianOnlyModeRadio = document.getElementById('gaussian-only-mode') as HTMLInputElement;
    this.localOnlyModeRadio = document.getElementById('local-only-mode') as HTMLInputElement;
    this.depthFusionModeRadio = document.getElementById('depth-fusion-mode') as HTMLInputElement;
    this.feedForwardModeRadio = document.getElementById('feed-forward-mode') as HTMLInputElement;

    // Debug controls
    this.depthDebugCheckbox = document.getElementById('depth-debug-checkbox') as HTMLInputElement;

    if (!this.wsConnectButton || !this.wsDisconnectButton) {
      debug.warn('[ControlPanel] Connection buttons not found');
    }
  }

  private setupListeners(): void {
    // Connection buttons
    this.wsConnectButton?.addEventListener('click', () => this.handleConnect());
    this.wsDisconnectButton?.addEventListener('click', () => this.handleDisconnect());

    // Encoder toggle
    this.jpegFallbackCheckbox?.addEventListener('click', () => this.handleEncoderChange());

    // Depth debug toggle
    this.depthDebugCheckbox?.addEventListener('click', () => this.handleDepthDebugToggle());

    // Render mode radios
    this.fusionModeRadio?.addEventListener('change', () => {
      if (this.fusionModeRadio?.checked) {
        this.handleRenderModeChange(RenderMode.FUSION);
      }
    });

    this.gaussianOnlyModeRadio?.addEventListener('change', () => {
      if (this.gaussianOnlyModeRadio?.checked) {
        this.handleRenderModeChange(RenderMode.GAUSSIAN_ONLY);
      }
    });

    this.localOnlyModeRadio?.addEventListener('change', () => {
      if (this.localOnlyModeRadio?.checked) {
        this.handleRenderModeChange(RenderMode.LOCAL_ONLY);
      }
    });

    this.depthFusionModeRadio?.addEventListener('change', () => {
      if (this.depthFusionModeRadio?.checked) {
        this.handleRenderModeChange(RenderMode.DEPTH_FUSION);
      }
    });

    this.feedForwardModeRadio?.addEventListener('change', () => {
      if (this.feedForwardModeRadio?.checked) {
        this.handleRenderModeChange(RenderMode.FEED_FORWARD);
      }
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
    const isJpegMode = this.jpegFallbackCheckbox?.checked ?? false;
    debug.logMain(`[ControlPanel] Encoder changed to ${isJpegMode ? 'JPEG' : 'H264'} mode`);
    this.callbacks.onEncoderChange?.(isJpegMode);
  }

  private handleDepthDebugToggle(): void {
    const isEnabled = this.depthDebugCheckbox?.checked ?? false;
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
    if (!this.wsStateConsoleText) return;

    const stateMap = {
      connected: 'WS State: Connected',
      disconnected: 'WS State: Disconnected',
      error: 'WS State: Error',
      closed: 'WS State: Closed'
    };

    this.wsStateConsoleText.textContent = stateMap[state];
  }

  getRenderMode(): RenderMode {
    return this.currentRenderMode;
  }

  setRenderMode(mode: RenderMode): void {
    this.currentRenderMode = mode;

    // Update radio buttons
    if (this.fusionModeRadio) this.fusionModeRadio.checked = (mode === RenderMode.FUSION);
    if (this.gaussianOnlyModeRadio) this.gaussianOnlyModeRadio.checked = (mode === RenderMode.GAUSSIAN_ONLY);
    if (this.localOnlyModeRadio) this.localOnlyModeRadio.checked = (mode === RenderMode.LOCAL_ONLY);
    if (this.depthFusionModeRadio) this.depthFusionModeRadio.checked = (mode === RenderMode.DEPTH_FUSION);
    if (this.feedForwardModeRadio) this.feedForwardModeRadio.checked = (mode === RenderMode.FEED_FORWARD);
  }

  isJpegMode(): boolean {
    return this.jpegFallbackCheckbox?.checked ?? false;
  }

  setJpegMode(enabled: boolean): void {
    if (this.jpegFallbackCheckbox) {
      this.jpegFallbackCheckbox.checked = enabled;
    }
  }

  isDepthDebugEnabled(): boolean {
    return this.depthDebugCheckbox?.checked ?? false;
  }

  setCallbacks(callbacks: ControlPanelCallbacks): void {
    this.callbacks = { ...this.callbacks, ...callbacks };
  }

  cleanup(): void {
    // Event listeners are automatically removed when elements are removed from DOM
  }
}

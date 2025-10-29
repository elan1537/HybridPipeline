/**
 * DebugPanel - Debug options and camera information display
 * Extracted from main.ts (lines 106-108, 126-143, 204-214, 372-420, 1640-1651)
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons';
import { debug } from '../../debug-logger';
import { CameraStateManager } from '../managers/CameraStateManager';

export interface DebugPanelCallbacks {
  onConsoleDebugToggle?: (enabled: boolean) => void;
  onCameraDebugToggle?: (enabled: boolean) => void;
  onSaveCamera?: () => void;
  onLoadCamera?: () => void;
}

export class DebugPanel {
  // DOM elements - Debug checkboxes
  private consoleDebugCheckbox: HTMLInputElement | null = null;
  private cameraDebugCheckbox: HTMLInputElement | null = null;

  // DOM elements - Camera info
  private cameraInfoSection: HTMLDivElement | null = null;
  private cameraPositionDiv: HTMLDivElement | null = null;
  private cameraTargetDiv: HTMLDivElement | null = null;
  private saveCameraButton: HTMLInputElement | null = null;
  private loadCameraButton: HTMLInputElement | null = null;

  // DOM elements - Manual camera control
  private cameraPosXInput: HTMLInputElement | null = null;
  private cameraPosYInput: HTMLInputElement | null = null;
  private cameraPosZInput: HTMLInputElement | null = null;
  private cameraTarXInput: HTMLInputElement | null = null;
  private cameraTarYInput: HTMLInputElement | null = null;
  private cameraTarZInput: HTMLInputElement | null = null;
  private applyCameraButton: HTMLInputElement | null = null;

  // Camera references (set externally)
  private camera: THREE.PerspectiveCamera | null = null;
  private controls: OrbitControls | null = null;

  private callbacks: DebugPanelCallbacks = {};

  constructor(callbacks: DebugPanelCallbacks = {}) {
    this.callbacks = callbacks;
    this.initializeElements();
    this.setupListeners();
  }

  private initializeElements(): void {
    // Debug checkboxes
    this.consoleDebugCheckbox = document.getElementById('console-debug-checkbox') as HTMLInputElement;
    this.cameraDebugCheckbox = document.getElementById('camera-debug-checkbox') as HTMLInputElement;

    // Camera info
    this.cameraInfoSection = document.getElementById('camera-info-section') as HTMLDivElement;
    this.cameraPositionDiv = document.getElementById('camera-position') as HTMLDivElement;
    this.cameraTargetDiv = document.getElementById('camera-target') as HTMLDivElement;
    this.saveCameraButton = document.getElementById('save-camera-button') as HTMLInputElement;
    this.loadCameraButton = document.getElementById('load-camera-button') as HTMLInputElement;

    // Manual camera control
    this.cameraPosXInput = document.getElementById('camera-pos-x') as HTMLInputElement;
    this.cameraPosYInput = document.getElementById('camera-pos-y') as HTMLInputElement;
    this.cameraPosZInput = document.getElementById('camera-pos-z') as HTMLInputElement;
    this.cameraTarXInput = document.getElementById('camera-tar-x') as HTMLInputElement;
    this.cameraTarYInput = document.getElementById('camera-tar-y') as HTMLInputElement;
    this.cameraTarZInput = document.getElementById('camera-tar-z') as HTMLInputElement;
    this.applyCameraButton = document.getElementById('apply-camera-button') as HTMLInputElement;
  }

  private setupListeners(): void {
    // Debug checkboxes
    this.consoleDebugCheckbox?.addEventListener('change', () => this.handleConsoleDebugToggle());
    this.cameraDebugCheckbox?.addEventListener('change', () => this.handleCameraDebugToggle());

    // Camera save/load
    this.saveCameraButton?.addEventListener('click', () => this.handleSaveCamera());
    this.loadCameraButton?.addEventListener('click', () => this.handleLoadCamera());

    // Manual camera control
    this.applyCameraButton?.addEventListener('click', () => this.applyCameraFromInputs());

    // Auto-apply after delay
    [this.cameraPosXInput, this.cameraPosYInput, this.cameraPosZInput,
     this.cameraTarXInput, this.cameraTarYInput, this.cameraTarZInput].forEach(input => {
      input?.addEventListener('input', () => {
        clearTimeout((input as any)._timeout);
        (input as any)._timeout = setTimeout(() => this.applyCameraFromInputs(), 500);
      });

      // Apply immediately on Enter
      input?.addEventListener('keydown', (event) => {
        if (event.key === 'Enter') {
          clearTimeout((input as any)._timeout);
          this.applyCameraFromInputs();
        }
      });
    });
  }

  // Event handlers
  private handleConsoleDebugToggle(): void {
    const isEnabled = this.consoleDebugCheckbox?.checked ?? false;
    debug.setDebugEnabled(isEnabled);
    debug.logMain(`Console debug logging ${isEnabled ? 'enabled' : 'disabled'}`);
    this.callbacks.onConsoleDebugToggle?.(isEnabled);
  }

  private handleCameraDebugToggle(): void {
    const isEnabled = this.cameraDebugCheckbox?.checked ?? false;
    this.callbacks.onCameraDebugToggle?.(isEnabled);
  }

  private handleSaveCamera(): void {
    if (!this.camera || !this.controls) {
      debug.warn('[DebugPanel] Camera not set');
      return;
    }
    CameraStateManager.save(this.camera, this.controls);
    this.callbacks.onSaveCamera?.();
  }

  private handleLoadCamera(): void {
    if (!this.camera || !this.controls) {
      debug.warn('[DebugPanel] Camera not set');
      return;
    }
    CameraStateManager.load(this.camera, this.controls);
    this.callbacks.onLoadCamera?.();
  }

  private applyCameraFromInputs(): void {
    if (!this.camera || !this.controls) return;

    const posX = parseFloat(this.cameraPosXInput?.value ?? '0') || 0;
    const posY = parseFloat(this.cameraPosYInput?.value ?? '0') || 0;
    const posZ = parseFloat(this.cameraPosZInput?.value ?? '0') || 0;
    const tarX = parseFloat(this.cameraTarXInput?.value ?? '0') || 0;
    const tarY = parseFloat(this.cameraTarYInput?.value ?? '0') || 0;
    const tarZ = parseFloat(this.cameraTarZInput?.value ?? '0') || 0;

    this.camera.position.set(posX, posY, posZ);
    this.controls.target.set(tarX, tarY, tarZ);
    this.controls.update();

    debug.logMain(`Camera applied: pos(${posX.toFixed(3)}, ${posY.toFixed(3)}, ${posZ.toFixed(3)}), target(${tarX.toFixed(3)}, ${tarY.toFixed(3)}, ${tarZ.toFixed(3)})`);
  }

  // Public methods
  setCameraReferences(camera: THREE.PerspectiveCamera, controls: OrbitControls): void {
    this.camera = camera;
    this.controls = controls;
  }

  updateCameraInfo(): void {
    if (!this.cameraDebugCheckbox?.checked) return;
    if (!this.camera || !this.controls) return;

    const position = this.camera.position;
    const target = this.controls.target;

    if (this.cameraPositionDiv) {
      this.cameraPositionDiv.textContent = `Position: (${position.x.toFixed(3)}, ${position.y.toFixed(3)}, ${position.z.toFixed(3)})`;
    }

    if (this.cameraTargetDiv) {
      this.cameraTargetDiv.textContent = `Target: (${target.x.toFixed(3)}, ${target.y.toFixed(3)}, ${target.z.toFixed(3)})`;
    }

    // Update input fields
    this.updateCameraInputFields();
  }

  private updateCameraInputFields(): void {
    if (!this.cameraDebugCheckbox?.checked) return;
    if (!this.camera || !this.controls) return;

    if (this.cameraPosXInput) this.cameraPosXInput.value = this.camera.position.x.toFixed(3);
    if (this.cameraPosYInput) this.cameraPosYInput.value = this.camera.position.y.toFixed(3);
    if (this.cameraPosZInput) this.cameraPosZInput.value = this.camera.position.z.toFixed(3);
    if (this.cameraTarXInput) this.cameraTarXInput.value = this.controls.target.x.toFixed(3);
    if (this.cameraTarYInput) this.cameraTarYInput.value = this.controls.target.y.toFixed(3);
    if (this.cameraTarZInput) this.cameraTarZInput.value = this.controls.target.z.toFixed(3);
  }

  isCameraDebugEnabled(): boolean {
    return this.cameraDebugCheckbox?.checked ?? false;
  }

  cleanup(): void {
    // Event listeners are automatically removed when elements are removed from DOM
  }
}

/**
 * DebugPanel - Debug options and camera information display
 * Extracted from main.ts (lines 106-108, 126-143, 204-214, 372-420, 1640-1651)
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons';
import { debug } from '../../debug-logger';
import { CameraStateManager } from '../managers/CameraStateManager';
import { BasePanel } from './BasePanel';

export interface DebugPanelCallbacks {
  onConsoleDebugToggle?: (enabled: boolean) => void;
  onCameraDebugToggle?: (enabled: boolean) => void;
  onSaveCamera?: () => void;
  onLoadCamera?: () => void;
}

export class DebugPanel extends BasePanel {
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
    super();
    this.callbacks = callbacks;
    this.initializeElements();
    this.setupListeners();
  }

  private initializeElements(): void {
    // Debug checkboxes
    this.consoleDebugCheckbox = this.getElement('console-debug-checkbox');
    this.cameraDebugCheckbox = this.getElement('camera-debug-checkbox');

    // Camera info
    this.cameraInfoSection = this.getElement('camera-info-section');
    this.cameraPositionDiv = this.getElement('camera-position');
    this.cameraTargetDiv = this.getElement('camera-target');
    this.saveCameraButton = this.getElement('save-camera-button');
    this.loadCameraButton = this.getElement('load-camera-button');

    // Manual camera control
    this.cameraPosXInput = this.getElement('camera-pos-x');
    this.cameraPosYInput = this.getElement('camera-pos-y');
    this.cameraPosZInput = this.getElement('camera-pos-z');
    this.cameraTarXInput = this.getElement('camera-tar-x');
    this.cameraTarYInput = this.getElement('camera-tar-y');
    this.cameraTarZInput = this.getElement('camera-tar-z');
    this.applyCameraButton = this.getElement('apply-camera-button');
  }

  private setupListeners(): void {
    // Debug checkboxes
    this.addListener(this.consoleDebugCheckbox, 'change', () => this.handleConsoleDebugToggle());
    this.addListener(this.cameraDebugCheckbox, 'change', () => this.handleCameraDebugToggle());

    // Camera save/load
    this.addListener(this.saveCameraButton, 'click', () => this.handleSaveCamera());
    this.addListener(this.loadCameraButton, 'click', () => this.handleLoadCamera());

    // Manual camera control
    this.addListener(this.applyCameraButton, 'click', () => this.applyCameraFromInputs());

    // Camera input fields - auto-apply after delay
    const cameraInputs = [
      this.cameraPosXInput, this.cameraPosYInput, this.cameraPosZInput,
      this.cameraTarXInput, this.cameraTarYInput, this.cameraTarZInput
    ];

    cameraInputs.forEach(input => {
      this.addListener(input, 'input', () => {
        clearTimeout((input as any)?._timeout);
        if (input) {
          (input as any)._timeout = setTimeout(() => this.applyCameraFromInputs(), 500);
        }
      });

      // Apply immediately on Enter
      this.addListener(input, 'keydown', (event) => {
        if (event.key === 'Enter') {
          clearTimeout((input as any)?._timeout);
          this.applyCameraFromInputs();
        }
      });
    });
  }

  // Event handlers
  private handleConsoleDebugToggle(): void {
    const isEnabled = this.isChecked(this.consoleDebugCheckbox);
    debug.setDebugEnabled(isEnabled);
    debug.logMain(`Console debug logging ${isEnabled ? 'enabled' : 'disabled'}`);
    this.callbacks.onConsoleDebugToggle?.(isEnabled);
  }

  private handleCameraDebugToggle(): void {
    const isEnabled = this.isChecked(this.cameraDebugCheckbox);
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

    const posX = parseFloat(this.getValue(this.cameraPosXInput, '0')) || 0;
    const posY = parseFloat(this.getValue(this.cameraPosYInput, '0')) || 0;
    const posZ = parseFloat(this.getValue(this.cameraPosZInput, '0')) || 0;
    const tarX = parseFloat(this.getValue(this.cameraTarXInput, '0')) || 0;
    const tarY = parseFloat(this.getValue(this.cameraTarYInput, '0')) || 0;
    const tarZ = parseFloat(this.getValue(this.cameraTarZInput, '0')) || 0;

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
    if (!this.isChecked(this.cameraDebugCheckbox)) return;
    if (!this.camera || !this.controls) return;

    const position = this.camera.position;
    const target = this.controls.target;

    this.updateText(
      this.cameraPositionDiv,
      `Position: (${position.x.toFixed(3)}, ${position.y.toFixed(3)}, ${position.z.toFixed(3)})`
    );

    this.updateText(
      this.cameraTargetDiv,
      `Target: (${target.x.toFixed(3)}, ${target.y.toFixed(3)}, ${target.z.toFixed(3)})`
    );

    // Update input fields
    this.updateCameraInputFields();
  }

  private updateCameraInputFields(): void {
    if (!this.isChecked(this.cameraDebugCheckbox)) return;
    if (!this.camera || !this.controls) return;

    this.setValue(this.cameraPosXInput, this.camera.position.x.toFixed(3));
    this.setValue(this.cameraPosYInput, this.camera.position.y.toFixed(3));
    this.setValue(this.cameraPosZInput, this.camera.position.z.toFixed(3));
    this.setValue(this.cameraTarXInput, this.controls.target.x.toFixed(3));
    this.setValue(this.cameraTarYInput, this.controls.target.y.toFixed(3));
    this.setValue(this.cameraTarZInput, this.controls.target.z.toFixed(3));
  }

  isCameraDebugEnabled(): boolean {
    return this.isChecked(this.cameraDebugCheckbox);
  }

  cleanup(): void {
    // Event listeners are automatically removed when elements are removed from DOM
  }
}

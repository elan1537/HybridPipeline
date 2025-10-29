/**
 * CameraStateManager - Manages camera position save/load via cookies
 * Extracted from main.ts (lines 236-277)
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons';
import { SettingsManager } from './SettingsManager';
import { debug } from '../../debug-logger';

interface CameraData {
  position: { x: number; y: number; z: number };
  target: { x: number; y: number; z: number };
}

export class CameraStateManager {
  private static readonly COOKIE_NAME = 'hybridpipeline_camera';

  /**
   * Save current camera position and target to cookie
   */
  static save(camera: THREE.PerspectiveCamera, controls: OrbitControls): void {
    const cameraData: CameraData = {
      position: {
        x: camera.position.x,
        y: camera.position.y,
        z: camera.position.z
      },
      target: {
        x: controls.target.x,
        y: controls.target.y,
        z: controls.target.z
      }
    };

    SettingsManager.setCookie(this.COOKIE_NAME, JSON.stringify(cameraData));
    debug.logMain(`Camera position saved: ${JSON.stringify(cameraData)}`);
  }

  /**
   * Load camera position and target from cookie
   * @returns true if loaded successfully, false otherwise
   */
  static load(camera: THREE.PerspectiveCamera, controls: OrbitControls): boolean {
    const cookieData = SettingsManager.getCookie(this.COOKIE_NAME);
    if (!cookieData) {
      debug.logMain('No saved camera position found');
      return false;
    }

    try {
      const cameraData: CameraData = JSON.parse(cookieData);

      // Set camera position
      camera.position.set(
        cameraData.position.x,
        cameraData.position.y,
        cameraData.position.z
      );

      // Set controls target
      controls.target.set(
        cameraData.target.x,
        cameraData.target.y,
        cameraData.target.z
      );

      // Update OrbitControls
      controls.update();

      debug.logMain(`Camera position loaded: ${JSON.stringify(cameraData)}`);
      return true;
    } catch (error) {
      debug.error('Failed to load camera position:', error);
      return false;
    }
  }

  /**
   * Clear saved camera position
   */
  static clear(): void {
    SettingsManager.deleteCookie(this.COOKIE_NAME);
    debug.logMain('Camera position cleared');
  }

  /**
   * Check if there's a saved camera position
   */
  static hasSaved(): boolean {
    return SettingsManager.hasCookie(this.COOKIE_NAME);
  }
}

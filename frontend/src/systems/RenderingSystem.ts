/**
 * RenderingSystem - Manages the main render loop
 * Moved from main.ts renderLoop() function
 */

import * as THREE from 'three';
import type { OrbitControls } from 'three/addons';
import type { System, SystemContext } from '../types';
import { debug } from '../debug-logger';

export enum RenderMode {
    FUSION = 'FUSION',
    GAUSSIAN_ONLY = 'GAUSSIAN_ONLY',
    LOCAL_ONLY = 'LOCAL_ONLY',
    DEPTH_FUSION = 'DEPTH_FUSION',
}

export interface RenderingConfig {
    // Scenes
    localScene: THREE.Scene;
    fusionScene: THREE.Scene;
    debugScene: THREE.Scene;
    gaussianOnlyScene: THREE.Scene;
    depthFusionScene: THREE.Scene;

    // Cameras
    camera: THREE.PerspectiveCamera;
    orthoCamera: THREE.OrthographicCamera;
    debugCamera: THREE.OrthographicCamera;
    gaussianOnlyCamera: THREE.OrthographicCamera;
    depthFusionCamera: THREE.OrthographicCamera;

    // Render targets
    localRenderTarget: THREE.WebGLRenderTarget;

    // Controls
    controls: OrbitControls;
    clock: THREE.Clock;

    // Callbacks
    onCameraUpdate?: () => void;
    onUpdate?: () => void;  // Called every frame

    // UI elements removed - now handled by UISystem
}

export class RenderingSystem implements System {
    name = 'rendering';

    private context!: SystemContext;
    private config!: RenderingConfig;
    private renderer!: THREE.WebGLRenderer;

    // State
    private currentRenderMode: RenderMode = RenderMode.FUSION;
    private frameCounter = 0;
    private currentTimeIndex = 0;
    private isPlaying = false;

    // FPS tracking
    private renderCnt = 0;
    private renderStart = 0;

    // Camera tracking
    private lastCameraUpdateTime = 0;
    private lastCameraPosition = new THREE.Vector3();
    private lastCameraTarget = new THREE.Vector3();
    private cameraUpdateInterval = 16.67; // ~60Hz

    async initialize(context: SystemContext): Promise<void> {
        this.context = context;
        this.renderer = context.renderingContext.renderer;
        debug.logMain('[RenderingSystem] Initialized');
    }

    /**
     * Configure the rendering system with scenes, cameras, etc.
     */
    configure(config: RenderingConfig): void {
        this.config = config;
        this.renderStart = performance.now();
        console.log('[RenderingSystem] Configured with onCameraUpdate:', !!config.onCameraUpdate);
    }

    /**
     * Set the current render mode
     */
    setRenderMode(mode: RenderMode): void {
        this.currentRenderMode = mode;
        debug.logMain(`[RenderingSystem] Render mode: ${mode}`);
    }

    /**
     * Update system state (called by Application.update())
     */
    update(deltaTime: number): void {
        if (!this.config) {
            console.warn('[RenderingSystem] update() called but not configured!');
            return;
        }

        // Update controls
        this.config.controls.update(deltaTime);

        // Update frame counter
        if (this.isPlaying) {
            this.frameCounter = (this.frameCounter + 1) % 300;
            this.currentTimeIndex = this.frameCounter / 299.0;
        }

        // FPS calculation
        const now = performance.now();
        const elapsed = now - this.renderStart;

        if (elapsed > 1000) {
            const duration = elapsed / 1000;
            const fps = this.renderCnt / duration;

            // FPS display removed - now handled by UISystem.updateRenderFPS()
            // This tracking is kept for internal monitoring only
            if (fps > 0 && fps <= 240 && isFinite(fps)) {
                debug.logMain(`[RenderingSystem] Render FPS: ${fps.toFixed(2)}`);
            }

            this.renderCnt = 0;
            this.renderStart = now;
        }

        // Camera update check
        const timeSinceLastUpdate = now - this.lastCameraUpdateTime;
        if (timeSinceLastUpdate > this.cameraUpdateInterval) {
            this.lastCameraPosition.copy(this.config.camera.position);
            this.lastCameraTarget.copy(this.config.controls.target);
            this.lastCameraUpdateTime = now;

            // Send camera frame to server
            if (this.config.onCameraUpdate) {
                console.log(`[RenderingSystem] Calling onCameraUpdate (interval=${timeSinceLastUpdate.toFixed(1)}ms)`);
                this.config.onCameraUpdate();
            } else {
                debug.logMain('[RenderingSystem] WARNING: onCameraUpdate callback not set!');
            }

            // Emit camera update event
            this.context.eventBus.emit('camera:update', {
                position: this.config.camera.position.clone(),
                target: this.config.controls.target.clone(),
            });
        }

        // Call custom update callback
        if (this.config.onUpdate) {
            this.config.onUpdate();
        }
    }

    /**
     * Execute rendering (called by Application.render())
     */
    render(): void {
        this.renderCnt++;
        // Depth debug checkbox removed - now controlled by currentRenderMode
        // If depth debug is needed, it should be added as a separate RenderMode
        {
            switch (this.currentRenderMode) {
                case RenderMode.FUSION:
                    this.renderer.setRenderTarget(this.config.localRenderTarget);
                    this.renderer.render(this.config.localScene, this.config.camera);
                    this.renderer.setRenderTarget(null);
                    this.renderer.render(this.config.fusionScene, this.config.orthoCamera);
                    break;

                case RenderMode.GAUSSIAN_ONLY:
                    this.renderer.render(
                        this.config.gaussianOnlyScene,
                        this.config.gaussianOnlyCamera
                    );
                    break;

                case RenderMode.LOCAL_ONLY:
                    this.renderer.render(this.config.localScene, this.config.camera);
                    break;

                case RenderMode.DEPTH_FUSION:
                    this.renderer.setRenderTarget(this.config.localRenderTarget);
                    this.renderer.render(this.config.localScene, this.config.camera);
                    this.renderer.setRenderTarget(null);
                    this.renderer.render(
                        this.config.depthFusionScene,
                        this.config.depthFusionCamera
                    );
                    break;
            }
        }
    }

    dispose(): void {
        debug.logMain('[RenderingSystem] Disposed');
    }
}
